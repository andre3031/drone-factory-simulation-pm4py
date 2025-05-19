import simpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# Scenario definitions with processing times
scenario_parameters = {
    "Scenario 1": {"setup": 2.0, "painting": 1.5, "assembly": 1.1},
    "Scenario 2": {"setup": 1.0, "painting": 1.5, "assembly": 1.1},
    "Scenario 3": {"setup": 2.0, "painting": 1.3, "assembly": 1.1},
    "Scenario 4": {"setup": 2.0, "painting": 1.5, "assembly": 1.0},
    "Scenario 5": {"setup": 1.0, "painting": 1.3, "assembly": 1.0}
}

# Loop through each scenario
for scenario_name, times in scenario_parameters.items():
    print(f"\n🔹 {scenario_name} 🔹")

    event_log = []  # Reset log for each scenario

    def log_event(product_id, event, time_ini, time_end, resource):
        event_log.append({
            'Product_id': product_id,
            'Machine_id': event,
            'Time_stamp_Ini': time_ini,
            'Time_stamp_Fim': time_end,
            'Simulated_time': time_end,
            'Resource': resource
        })

    class DroneFactory:
        def __init__(self, env):
            self.plastic = simpy.Container(env, capacity=1000, init=500)
            self.electronic = simpy.Container(env, capacity=100, init=100)
            self.first_body_buffer = simpy.Container(env, capacity=100, init=0)
            self.first_propeller_buffer = simpy.Container(env, capacity=100, init=0)
            self.second_body_buffer = simpy.Container(env, capacity=200, init=0)
            self.second_propeller_buffer = simpy.Container(env, capacity=200, init=0)
            self.dispatch = simpy.Container(env, capacity=500, init=0)

    def body_maker(env, factory):
        product_id = 1
        while True:
            start = env.now
            log_event(product_id, 'Start Body Making', start, None, 'body_maker')
            yield factory.plastic.get(1)
            yield env.timeout(1)
            yield factory.first_body_buffer.put(1)
            end = env.now
            log_event(product_id, 'End Body Making', start, end, 'body_maker')
            product_id += 1

    def propeller_maker(env, factory):
        product_id = 1
        while True:
            start = env.now
            log_event(product_id, 'Start Propeller Making', start, None, 'propeller_maker')
            yield factory.plastic.get(1)
            yield env.timeout(1)
            yield factory.first_propeller_buffer.put(4)
            end = env.now
            log_event(product_id, 'End Propeller Making', start, end, 'propeller_maker')
            product_id += 1

    def painter(env, factory):
        product_id = 1
        while True:
            start = env.now
            log_event(product_id, 'Start Painting', start, None, 'painter')
            yield factory.first_body_buffer.get(2)
            yield factory.first_propeller_buffer.get(8)
            yield env.timeout(times["painting"])
            yield factory.second_propeller_buffer.put(8)
            yield factory.second_body_buffer.put(2)
            end = env.now
            log_event(product_id, 'End Painting', start, end, 'painter')
            product_id += 1

    def assembler(env, factory):
        product_id = 1
        while True:
            start = env.now
            log_event(product_id, 'Start Assembling', start, None, 'assembler')
            yield factory.second_propeller_buffer.get(4)
            yield factory.second_body_buffer.get(1)
            yield factory.electronic.get(1)
            yield env.timeout(times["assembly"])
            yield factory.dispatch.put(1)
            end = env.now
            log_event(product_id, 'End Assembling', start, end, 'assembler')
            product_id += 1

    # Run simulation
    env = simpy.Environment()
    factory = DroneFactory(env)
    env.process(body_maker(env, factory))
    env.process(propeller_maker(env, factory))
    env.process(painter(env, factory))
    env.process(assembler(env, factory))
    env.run(until=40)

    # Create DataFrame
    event_log_df = pd.DataFrame(event_log)

    # Scrollable HTML table
    html_table = event_log_df.to_html(index=False)
    scrollable_html = f"""
    <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 5px;">
        {html_table}
    </div>
    """
    display(HTML(scrollable_html))

    # Plot: Number of events per resource
    plt.figure(figsize=(8, 4))
    sns.countplot(data=event_log_df, x='Resource', order=event_log_df['Resource'].value_counts().index)
    plt.title(f"Number of events per resource - {scenario_name}")
    plt.xlabel("Resource")
    plt.ylabel("Number of events")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# === PM4Py ANALYSIS ===
# Convert to PM4Py format
log_pm4py = event_log_df.copy()

log_pm4py = log_pm4py.rename(columns={
    'Product_id': 'case:concept:name',
    'Machine_id': 'concept:name',
    'Simulated_time': 'time:timestamp'
})

log_pm4py['time:timestamp'] = pd.to_datetime(log_pm4py['time:timestamp'], unit='h', origin=pd.Timestamp('2024-01-01'))

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

log_pm4py = dataframe_utils.convert_timestamp_columns_in_df(log_pm4py)
event_log = log_converter.apply(log_pm4py)

# Heuristic Miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

heu_net = heuristics_miner.apply_heu(event_log)
gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)

# Conformance checking
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.visualization.petri_net import visualizer as pn_visualizer

net, initial_marking, final_marking = hn_converter.apply(heu_net)
replayed_traces = token_replay.apply(event_log, net, initial_marking, final_marking)

fitness_values = [trace['trace_fitness'] for trace in replayed_traces if trace['trace_is_fit']]
mean_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0

print(f"✔️ Mean fitness of conforming traces: {mean_fitness:.2f}")

# Save Petri net image
from IPython.display import Image, display
output_path = "/content/petri_net_conformance.png"
gviz_petri = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.save(gviz_petri, output_path)
display(Image(filename=output_path))
print(f"📁 Petri net image saved at: {output_path}")
