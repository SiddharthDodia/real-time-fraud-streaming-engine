import json
import time
from collections import deque
from confluent_kafka import Consumer, KafkaException, KafkaError
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.console import Group

# Store recent transactions to show in a rolling window
recent_txs = deque(maxlen=15)

# Rolling stats
stats = {
    "total": 0,
    "true_positives": 0,
    "true_negatives": 0,
    "false_positives": 0,
    "false_negatives": 0,
    "total_latency": 0.0
}

def create_dashboard():
    # 1. Recent Transactions Table
    tx_table = Table(box=box.MINIMAL_DOUBLE_HEAD, expand=True)
    tx_table.add_column("CC Num", style="cyan")
    tx_table.add_column("Amount", style="green")
    tx_table.add_column("Model Prob", justify="right")
    tx_table.add_column("Decision", justify="center")
    tx_table.add_column("Actual", justify="center")
    tx_table.add_column("Status", justify="center")
    tx_table.add_column("Latency (ms)", justify="right", style="magenta")
    
    for tx in recent_txs:
        # Determine formatting based on status
        if tx['status'] == "CORRECT (TP)":
            status_str = "[bold green]CAUGHT FRAUD[/bold green]"
        elif tx['status'] == "CORRECT (TN)":
            status_str = "[green]VALID[/green]"
        elif tx['status'] == "FALSE POSITIVE":
            status_str = "[bold yellow]FALSE ALARM[/bold yellow]"
        elif tx['status'] == "FALSE NEGATIVE":
            status_str = "[bold red]MISSED FRAUD![/bold red]"
        else:
            status_str = tx['status']
            
        prob_color = "red" if tx['prob'] > 0.85 else "green"
        
        tx_table.add_row(
            tx['cc_num'][-4:], # Mask cc num, show last 4
            f"${tx['amt']:.2f}",
            f"[{prob_color}]{tx['prob']:.4f}[/{prob_color}]",
            tx['decision'],
            "FRAUD" if tx['actual'] == 1 else "LEGIT",
            status_str,
            f"{tx['latency']:.1f}ms"
        )
        
    # 2. Stats Panel
    if stats["total"] > 0:
        precision = stats["true_positives"] / (stats["true_positives"] + stats["false_positives"]) if (stats["true_positives"] + stats["false_positives"]) > 0 else 0
        recall = stats["true_positives"] / (stats["true_positives"] + stats["false_negatives"]) if (stats["true_positives"] + stats["false_negatives"]) > 0 else 0
        avg_latency = stats["total_latency"] / stats["total"]
    else:
        precision = 0
        recall = 0
        avg_latency = 0
        
    stats_text = (
        f"Total Processed: [bold cyan]{stats['total']}[/bold cyan] | "
        f"Precision: [bold green]{precision:.2f}[/bold green] | "
        f"Recall: [bold green]{recall:.2f}[/bold green] | "
        f"Avg Latency: [bold magenta]{avg_latency:.1f}ms[/bold magenta]\n"
        f"True Positives: [green]{stats['true_positives']}[/green] | "
        f"True Negatives: [green]{stats['true_negatives']}[/green] | "
        f"False Positives: [yellow]{stats['false_positives']}[/yellow] | "
        f"False Negatives: [red]{stats['false_negatives']}[/red]"
    )
    
    group = Group(
        Panel(stats_text, title="Real-Time Metrics", border_style="blue"),
        Panel(tx_table, title="Live Transaction Stream", border_style="cyan")
    )
    
    return group

def process_message(msg):
    try:
        data = json.loads(msg.value().decode('utf-8'))
        
        is_fraud_true = data.get("is_fraud_true")
        decision = data.get("decision")
        latency_ms = data.get("latency_ms", 0.0)
        
        status = "UNKNOWN"
        if is_fraud_true is not None:
            if decision == "DECLINE" and is_fraud_true == 1:
                stats["true_positives"] += 1
                status = "CORRECT (TP)"
            elif decision == "APPROVE" and is_fraud_true == 0:
                stats["true_negatives"] += 1
                status = "CORRECT (TN)"
            elif decision == "DECLINE" and is_fraud_true == 0:
                stats["false_positives"] += 1
                status = "FALSE POSITIVE"
            elif decision == "APPROVE" and is_fraud_true == 1:
                stats["false_negatives"] += 1
                status = "FALSE NEGATIVE"
        
        stats["total"] += 1
        stats["total_latency"] += latency_ms
        
        recent_txs.append({
            "cc_num": data.get("cc_num", "----"),
            "amt": data.get("amt", 0.0),
            "prob": data.get("fraud_probability", 0.0),
            "decision": decision,
            "actual": is_fraud_true,
            "status": status,
            "latency": latency_ms
        })
    except Exception as e:
        pass

def main():
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'dashboard_group',
        'auto.offset.reset': 'latest' # Only show new transactions
    }
    consumer = Consumer(conf)
    consumer.subscribe(['scored-decisions'])

    print("Connecting to Kafka...")
    
    with Live(create_dashboard(), refresh_per_second=4) as live:
        try:
            while True:
                msg = consumer.poll(timeout=0.1)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        break
                
                process_message(msg)
                live.update(create_dashboard())
                
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()

if __name__ == "__main__":
    main()
