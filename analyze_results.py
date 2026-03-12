import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_FILE = "batch_run_results.json"
OUTPUT_CSV = "interview_metrics.csv"
OUTPUT_TOPICS_CHART = "topics_covered_chart.png"
OUTPUT_FACTS_CHART = "facts_extracted_chart.png"
OUTPUT_FLAGS_CHART = "flags_chart.png"
OUTPUT_STATUS_CHART = "status_pie_chart.png"


def load_results(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_rows(results):
    rows = []
    for item in results:
        rows.append(
            {
                "persona_id": item.get("persona_id"),
                "name": item.get("name"),
                "status": item.get("status"),
                "completed": item.get("completed"),
                "turns_run": item.get("turns_run", 0),
                "turns_recorded": item.get("turns_recorded", 0),
                "covered_topics_count": len(item.get("covered_topics", [])),
                "facts_count": len(item.get("facts", {})),
                "timeline_count": len(item.get("timeline", [])),
                "flags_count": len(item.get("flags", [])),
            }
        )
    return rows


def save_csv(rows, path: str):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(labels, values, title, ylabel, output_path):
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Persona")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def pie_chart(status_counts, output_path):
    labels = list(status_counts.keys())
    values = list(status_counts.values())

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title("Interview Status Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    results = load_results(INPUT_FILE)
    rows = build_rows(results)
    save_csv(rows, OUTPUT_CSV)

    labels = [row["persona_id"] for row in rows]
    topics = [row["covered_topics_count"] for row in rows]
    facts = [row["facts_count"] for row in rows]
    flags = [row["flags_count"] for row in rows]

    bar_chart(
        labels,
        topics,
        "Topics Covered by Persona",
        "Number of Topics Covered",
        OUTPUT_TOPICS_CHART,
    )

    bar_chart(
        labels,
        facts,
        "Facts Extracted by Persona",
        "Number of Facts Extracted",
        OUTPUT_FACTS_CHART,
    )

    bar_chart(
        labels,
        flags,
        "Flags Raised by Persona",
        "Number of Flags",
        OUTPUT_FLAGS_CHART,
    )

    status_counts = {}
    for row in rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    pie_chart(status_counts, OUTPUT_STATUS_CHART)

    print("Analysis complete.")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved chart: {OUTPUT_TOPICS_CHART}")
    print(f"Saved chart: {OUTPUT_FACTS_CHART}")
    print(f"Saved chart: {OUTPUT_FLAGS_CHART}")
    print(f"Saved chart: {OUTPUT_STATUS_CHART}")


if __name__ == "__main__":
    main()