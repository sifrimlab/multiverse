import asyncio
import random
from rich.live import Live
from rich.table import Table
from rich.console import Console

def generate_table(tasks) -> Table:
    table = Table(title="Multiverse Parallel Tasks")
    table.add_column("Task Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", style="magenta")

    for name, status in tasks.items():
        style = "green" if status == "Success" or status == "Ready" else "red" if "Failed" in status or "Error" in status else "yellow"
        table.add_row(name, f"[{style}]{status}[/{style}]")

    return table

async def simulate_task(name, tasks, min_time=2, max_time=5):
    tasks[name] = "Pending"
    await asyncio.sleep(random.uniform(0.5, 1.5))

    tasks[name] = "Building/Pulling"
    await asyncio.sleep(random.uniform(min_time, max_time))

    tasks[name] = "Running"
    await asyncio.sleep(random.uniform(min_time, max_time))

    if random.random() > 0.1:
        tasks[name] = "Success"
    else:
        tasks[name] = "Failed (1)"

async def main():
    tasks = {f"Model_{i}": "Queued" for i in range(1, 6)}

    with Live(generate_table(tasks), refresh_per_second=4) as live:
        simulations = [simulate_task(name, tasks) for name in tasks.keys()]
        for _ in range(20):
            await asyncio.sleep(0.25)
            live.update(generate_table(tasks))
        await asyncio.gather(*simulations)
        live.update(generate_table(tasks))

if __name__ == "__main__":
    asyncio.run(main())
