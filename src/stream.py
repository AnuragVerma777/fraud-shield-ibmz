"""
stream.py — Real-time event streaming simulation for Fraud Shield.

Mimics a Kafka-style event streaming pipeline using Python's queue.Queue:
  1. Producer Thread:
     - Reads rows from the TEST set in chronological (Time) order.
     - Builds transaction event dictionaries.
     - Controls emission speed using fixed transactions-per-second (TPS)
       or a realtime_factor (compressing real-world Time deltas).
     - Pushes events to a thread-safe Queue (acting as Kafka topic).
  2. Consumer Thread:
     - Pulls events from the queue.
     - Sends feature vectors to RiskEngine.score_transaction() (keeping
       the true ground-truth label hidden from the scoring model).
     - Produces scoring results (decision, probabilities, latencies).
     - Appends results immediately to data/stream_log.csv with immediate
       flushing for live dashboard visualization.

Usage:
    python -m src.stream                     # Run default demo (2,000 events, 20 TPS)
    python -m src.stream --max-events 500    # Run 500 events
    python -m src.stream --tps 50            # Run at 50 transactions/second
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import os
import queue
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    ALERTS_PATH,
    PCA_FEATURES,
    RAW_FEATURES,
    STREAM_LOG_PATH,
    ensure_dirs,
)
from src.data import get_splits
from src.risk import RiskEngine
from src.alerts import ALERT_FIELDNAMES, generate_alert

# Sentinel marker placed on the queue to signal to the consumer that
# production has concluded and the consumer thread should exit cleanly.
_SENTINEL = object()

# Column headers for the streaming output CSV file
CSV_FIELDNAMES = [
    "txn_id",
    "risk_score",
    "decision",
    "xgb_prob",
    "anomaly_score",
    "latency_ms",
    "raw_amount",
    "true_label",
    "processed_at",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA PREPARATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def load_stream_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load test set transactions sorted in their original chronological order.

    The model was trained on scaled data, but RiskEngine.score_transaction()
    expects raw, unscaled 'Time' and 'Amount' features because it applies its
    own fitted scaler internally.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with canonical feature columns (V1-V28, Time, Amount)
        in raw form, ordered by original transaction time.
    labels_series : pd.Series
        Ground-truth labels (0 for genuine, 1 for fraud) in the same order.
    """
    # Load stratified train/val/test splits
    _, _, X_test, _, _, y_test = get_splits()

    # Sort the test set rows strictly by original unscaled 'Time'
    # to accurately replay chronological transaction flow.
    sort_order = X_test["Time_original"].argsort()
    sorted_features = X_test.iloc[sort_order].copy()
    sorted_labels = y_test.iloc[sort_order].copy()

    # Restore raw unscaled Time and Amount into canonical feature names
    sorted_features["Time"] = sorted_features["Time_original"]
    sorted_features["Amount"] = sorted_features["Amount_original"]

    return sorted_features[ALL_FEATURES], sorted_labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRODUCER THREAD (mimics Kafka Producer / Message Broker)
# ─────────────────────────────────────────────────────────────────────────────
class TransactionProducer(threading.Thread):
    """
    Thread that emits transaction events into an in-memory queue.

    Parameters
    ----------
    event_queue : queue.Queue
        Shared thread-safe queue holding events.
    features_df : pd.DataFrame
        Test transactions with unscaled feature columns.
    labels_series : pd.Series
        Ground-truth fraud labels (0 or 1).
    transactions_per_second : float
        Pacing rate when realtime_factor is None (default 20.0).
    realtime_factor : float | None
        Speedup factor for real time gaps (e.g. 100.0 means 100x faster than real-life).
    max_events : int | None
        Upper limit on transactions to emit (None for full test set).
    stop_event : threading.Event | None
        Flag for cooperative cancellation.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        features_df: pd.DataFrame,
        labels_series: pd.Series,
        transactions_per_second: float = 20.0,
        realtime_factor: Optional[float] = None,
        max_events: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(name="ProducerThread", daemon=True)
        self.queue = event_queue
        self.features_df = features_df
        self.labels_series = labels_series
        self.transactions_per_second = float(transactions_per_second)
        self.realtime_factor = (
            float(realtime_factor) if realtime_factor is not None else None
        )
        self.max_events = max_events
        self.stop_event = stop_event or threading.Event()
        self.events_produced = 0

    def stop(self) -> None:
        """Signal the producer to stop sending events."""
        self.stop_event.set()

    def run(self) -> None:
        """Main loop: iterate through test rows and publish events to queue."""
        total_rows = len(self.features_df)
        limit = total_rows if self.max_events is None else min(total_rows, self.max_events)

        prev_event_time: Optional[float] = None

        # Pre-extract values for faster iteration over pandas data
        feature_dicts = self.features_df.to_dict(orient="records")
        row_indices = self.features_df.index.tolist()
        label_values = self.labels_series.to_numpy()

        for i in range(limit):
            # Check if an external cancellation has been requested (e.g. Ctrl+C)
            if self.stop_event.is_set():
                break

            row_features = feature_dicts[i]
            txn_idx = row_indices[i]
            raw_time = float(row_features["Time"])
            raw_amount = float(row_features["Amount"])
            true_label = int(label_values[i])

            # Calculate pacing delay between consecutive events
            delay_sec = 0.0
            if self.realtime_factor is not None and self.realtime_factor > 0:
                if prev_event_time is not None:
                    delta_t = raw_time - prev_event_time
                    delay_sec = max(0.0, delta_t / self.realtime_factor)
                prev_event_time = raw_time
            else:
                if self.transactions_per_second > 0:
                    delay_sec = 1.0 / self.transactions_per_second

            # Sleep cooperatively: if stop_event is signaled during wait, break immediately
            if delay_sec > 0 and i > 0:
                interrupted = self.stop_event.wait(timeout=delay_sec)
                if interrupted:
                    break

            # Assemble event payload:
            # - txn_id: unique transaction identifier
            # - event_time: original transaction time in seconds
            # - raw_amount: monetary amount
            # - true_label: actual fraud status (0 or 1), kept separate from features
            # - features: feature dictionary supplied to the model
            event = {
                "txn_id": f"tx_{txn_idx}",
                "event_time": raw_time,
                "raw_amount": raw_amount,
                "true_label": true_label,
                "features": row_features,
            }

            self.queue.put(event)
            self.events_produced += 1

        # Signal consumer that stream has ended
        self.queue.put(_SENTINEL)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSUMER THREAD (Scoring Engine + Live Stream Logger)
# ─────────────────────────────────────────────────────────────────────────────
class TransactionConsumer(threading.Thread):
    """
    Thread that pulls events from the queue, scores them, and logs results.

    Parameters
    ----------
    event_queue : queue.Queue
        Shared thread-safe queue holding events.
    risk_engine : RiskEngine
        Scoring engine instance.
    log_path : str
        Path to append-only CSV output file.
    stop_event : threading.Event | None
        Cooperative stop signal.
    progress_interval : int
        Print a progress line every N events.
    progress_callback : Callable[[int, dict], None] | None
        Optional hook invoked on each progress interval.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        risk_engine: RiskEngine,
        log_path: str = STREAM_LOG_PATH,
        alerts_path: str = ALERTS_PATH,
        stop_event: Optional[threading.Event] = None,
        progress_interval: int = 200,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ):
        super().__init__(name="ConsumerThread", daemon=True)
        self.queue = event_queue
        self.engine = risk_engine
        self.log_path = log_path
        self.alerts_path = alerts_path
        self.stop_event = stop_event or threading.Event()
        self.progress_interval = progress_interval
        self.progress_callback = progress_callback

        self.events_processed = 0
        self.decisions = Counter()
        self.latencies: list[float] = []
        self.start_time: float = 0.0

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self.stop_event.set()

    def run(self) -> None:
        """Main loop: dequeue events, score them, and append to CSV."""
        ensure_dirs()
        log_dir = os.path.dirname(os.path.abspath(self.log_path))
        os.makedirs(log_dir, exist_ok=True)

        # Check if CSV header is needed
        write_header = (not os.path.exists(self.log_path)) or (
            os.path.getsize(self.log_path) == 0
        )

        self.start_time = time.perf_counter()

        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
                f.flush()

            while not self.stop_event.is_set():
                try:
                    # Timeout periodically to re-evaluate self.stop_event
                    item = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Sentinel signifies end of producer stream
                if item is _SENTINEL:
                    self.queue.task_done()
                    break

                event: dict = item

                # True label is kept hidden from scorer; pass only the feature vector!
                feature_input = event["features"]
                score_output = self.engine.score_transaction(feature_input)

                # Compose the final stream record
                processed_time = datetime.now(timezone.utc).isoformat()
                result = {
                    "txn_id": event["txn_id"],
                    "risk_score": score_output["risk_score"],
                    "decision": score_output["decision"],
                    "xgb_prob": score_output["xgb_prob"],
                    "anomaly_score": score_output["anomaly_score"],
                    "latency_ms": score_output["latency_ms"],
                    "raw_amount": event["raw_amount"],
                    "true_label": event["true_label"],
                    "processed_at": processed_time,
                }

                # Write immediately and flush for live reader visibility
                writer.writerow(result)
                f.flush()

                # Persist an explainable rule-based alert for FLAG/BLOCK.
                alert = generate_alert(result)
                if alert is not None:
                    self._append_alert(alert)

                # Bookkeeping
                self.events_processed += 1
                self.decisions[result["decision"]] += 1
                self.latencies.append(result["latency_ms"])
                self.queue.task_done()

                # Report progress
                if (
                    self.progress_interval > 0
                    and self.events_processed % self.progress_interval == 0
                ):
                    self._report_progress(result)

    def _append_alert(self, alert: dict) -> None:
        """Append one alert row, writing a header for a new/empty CSV."""
        alerts_dir = os.path.dirname(os.path.abspath(self.alerts_path))
        os.makedirs(alerts_dir, exist_ok=True)
        write_header = (not os.path.exists(self.alerts_path)) or (
            os.path.getsize(self.alerts_path) == 0
        )
        with open(self.alerts_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALERT_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(alert)
            f.flush()

    def _report_progress(self, latest_result: dict) -> None:
        """Print an informative single-line progress update."""
        elapsed = max(time.perf_counter() - self.start_time, 1e-4)
        tps = self.events_processed / elapsed
        recent_latencies = self.latencies[-self.progress_interval :]
        avg_latency = float(np.mean(recent_latencies)) if recent_latencies else 0.0

        line = (
            f"[Stream] {self.events_processed:>5,d} events | "
            f"Rate: {tps:5.1f} evt/s | "
            f"Avg Latency: {avg_latency:5.2f} ms | "
            f"Decisions: APPROVE={self.decisions['APPROVE']}, "
            f"FLAG={self.decisions['FLAG']}, "
            f"BLOCK={self.decisions['BLOCK']}"
        )
        print(line)

        if self.progress_callback is not None:
            self.progress_callback(self.events_processed, latest_result)


# ─────────────────────────────────────────────────────────────────────────────
# 4. STREAM SIMULATOR COORDINATOR
# ─────────────────────────────────────────────────────────────────────────────
class StreamSimulator:
    """
    Coordinating manager for the Kafka-style producer and consumer threads.

    Supports start, stop, and max_events controls with clean shutdown.
    """

    def __init__(
        self,
        features_df: Optional[pd.DataFrame] = None,
        labels_series: Optional[pd.Series] = None,
        transactions_per_second: float = 20.0,
        realtime_factor: Optional[float] = None,
        max_events: Optional[int] = 2000,
        log_path: str = STREAM_LOG_PATH,
        alerts_path: str = ALERTS_PATH,
        risk_engine: Optional[RiskEngine] = None,
        progress_interval: int = 200,
        queue_maxsize: int = 2000,
    ):
        self.features_df = features_df
        self.labels_series = labels_series
        self.transactions_per_second = transactions_per_second
        self.realtime_factor = realtime_factor
        self.max_events = max_events
        self.log_path = log_path
        self.alerts_path = alerts_path
        self.risk_engine = risk_engine
        self.progress_interval = progress_interval
        self.queue_maxsize = queue_maxsize

        self.queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.stop_event: threading.Event = threading.Event()
        self.producer: Optional[TransactionProducer] = None
        self.consumer: Optional[TransactionConsumer] = None

    def start(self) -> None:
        """Initialize data, models, and start producer & consumer threads."""
        self.stop_event.clear()

        # Load dataset if not provided
        if self.features_df is None or self.labels_series is None:
            print("[StreamSimulator] Loading chronological test set...")
            self.features_df, self.labels_series = load_stream_dataset()

        # Initialize RiskEngine if not provided
        if self.risk_engine is None:
            print("[StreamSimulator] Initializing RiskEngine...")
            self.risk_engine = RiskEngine()

        # Instantiate threads
        self.producer = TransactionProducer(
            event_queue=self.queue,
            features_df=self.features_df,
            labels_series=self.labels_series,
            transactions_per_second=self.transactions_per_second,
            realtime_factor=self.realtime_factor,
            max_events=self.max_events,
            stop_event=self.stop_event,
        )

        self.consumer = TransactionConsumer(
            event_queue=self.queue,
            risk_engine=self.risk_engine,
            log_path=self.log_path,
            alerts_path=self.alerts_path,
            stop_event=self.stop_event,
            progress_interval=self.progress_interval,
        )

        self.consumer.start()
        self.producer.start()
        print(
            f"[StreamSimulator] Stream started: max_events={self.max_events}, "
            f"TPS={self.transactions_per_second}, realtime_factor={self.realtime_factor}, "
            f"log={self.log_path}"
        )

    def stop(self, timeout: float = 3.0) -> None:
        """Signal threads to stop and cleanly wait for termination."""
        self.stop_event.set()
        # Put sentinel to unblock consumer if producer hadn't reached end
        try:
            self.queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass

        if self.producer and self.producer.is_alive():
            self.producer.join(timeout=timeout)
        if self.consumer and self.consumer.is_alive():
            self.consumer.join(timeout=timeout)
        print("[StreamSimulator] Stream stopped cleanly.")

    def is_alive(self) -> bool:
        """Check if either producer or consumer is still running."""
        prod_alive = bool(self.producer and self.producer.is_alive())
        cons_alive = bool(self.consumer and self.consumer.is_alive())
        return prod_alive or cons_alive

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for both threads to finish execution."""
        if self.producer:
            self.producer.join(timeout=timeout)
        if self.consumer:
            self.consumer.join(timeout=timeout)

    def run(self) -> None:
        """
        Execute the stream synchronously and handle Ctrl+C cleanly.
        """
        self.start()
        try:
            while self.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[StreamSimulator] Interrupted by user (Ctrl+C). Shutting down...")
            self.stop()
            sys.exit(0)

        # Print final summary stats
        if self.consumer:
            total = self.consumer.events_processed
            dec = self.consumer.decisions
            avg_lat = (
                float(np.mean(self.consumer.latencies))
                if self.consumer.latencies
                else 0.0
            )
            print("\n" + "=" * 60)
            print("STREAM SIMULATION COMPLETE")
            print(f"Total Transactions Processed: {total:,d}")
            print(f"Decisions Breakdown:")
            for d in ["APPROVE", "FLAG", "BLOCK"]:
                cnt = dec[d]
                pct = (cnt / total * 100) if total > 0 else 0.0
                print(f"  - {d:<8}: {cnt:>5,d} ({pct:5.1f}%)")
            print(f"Average Inference Latency  : {avg_lat:.2f} ms")
            print(f"Output Written To          : {self.log_path}")
            print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI & MAIN BLOCK
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-time transaction event stream simulation."
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=2000,
        help="Number of transaction events to run (default: 2000).",
    )
    parser.add_argument(
        "--tps",
        type=float,
        default=20.0,
        help="Transactions per second speed (default: 20.0).",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=None,
        help="Compression factor for real-world Time deltas (e.g. 100).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=200,
        help="Print progress every N events (default: 200).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=STREAM_LOG_PATH,
        help=f"Path to stream log file (default: {STREAM_LOG_PATH}).",
    )
    parser.add_argument(
        "--alerts-path",
        type=str,
        default=ALERTS_PATH,
        help=f"Path to alert CSV (default: {ALERTS_PATH}).",
    )
    parser.add_argument(
        "--reset-log",
        action="store_true",
        help="Remove existing stream log before starting.",
    )
    args = parser.parse_args()

    if args.reset_log and os.path.exists(args.log_path):
        os.remove(args.log_path)
        print(f"[Stream] Removed existing log file: {args.log_path}")

    simulator = StreamSimulator(
        max_events=args.max_events,
        transactions_per_second=args.tps,
        realtime_factor=args.realtime_factor,
        progress_interval=args.progress_interval,
        log_path=args.log_path,
        alerts_path=args.alerts_path,
    )
    simulator.run()


if __name__ == "__main__":
    main()
