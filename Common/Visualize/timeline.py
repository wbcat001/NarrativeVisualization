import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


# df = pd.DataFrame([
#     dict(Task="Job A", Start='0', Finish='10', Resource="Alex"),
#     dict(Task="Job B", Start='10', Finish='20', Resource="Alex"),
#     dict(Task="Job C", Start='5', Finish='15', Resource="Max")
# ])

# fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Resource")
# fig.show()

import plotly.figure_factory as ff

from datetime import datetime, timedelta
import numpy as np

def convert_to_datetime(x):
  result = datetime(1971, 1, 1) + timedelta(days=int(x))

  return result.strftime("%Y-%m-%d")


def draw_gantt(df, index_col, colors=None):
  min_time = min(df["Start"])
  max_time = max(df["Finish"])
  df["Start"] = df["Start"].apply(convert_to_datetime)
  df["Finish"] = df["Finish"].apply(convert_to_datetime)



  num_tick_labels = np.linspace(start = min_time, stop = max_time, num = max_time - min_time + 1, dtype = int)

  date_ticks = [convert_to_datetime(x) for x in num_tick_labels]
  if colors:
    fig = ff.create_gantt(df, group_tasks=True, index_col=index_col, colors=colors)
  else:
    fig = ff.create_gantt(df, group_tasks=True, index_col=index_col)

  fig.layout.xaxis.update({
          'tickvals' : date_ticks,
          'ticktext' : num_tick_labels,
          })
  fig.show() #.write_html('first_figure.html', auto_open=True)




if __name__ == "__main__":
  data = [dict(Task="Job A", Start=0, Finish=4),
      dict(Task="Job B", Start=3, Finish=6),
      dict(Task="Job C", Start=6, Finish=10),
      dict(Task="Job A", Start=17, Finish=20),
      dict(Task="Job C", Start=19, Finish=30)]
  df = pd.DataFrame(data)
  draw_gantt(df, "Task")