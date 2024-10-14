import plotly.express as px
from vectordb_bench.frontend.config.styles import COLOR_MAP
from vectordb_bench.metric import metricUnitMap


def drawCharts(st, allData, failedTasks, _):
    dataset_names = list(set([data["dataset_name"] for data in allData]))
    dataset_names.sort()
    for dataset_name in dataset_names:
        container = st.container()
        container.subheader(dataset_name)
        data = [d for d in allData if d["dataset_name"] == dataset_name]
        drawChartByMetric(container, data)


def drawChartByMetric(st, data):
    metrics = ["qps", "recall"]
    columns = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        container = columns[i]
        container.markdown(f"#### {metric}")
        drawChart(container, data, metric)

    showMetric(st.container(), "load_duration", "db_name", data)


def showMetric(st, key: str, group_by: str, data):
    st.markdown(f"#### {key}")
    item_names = list(set([d[group_by] for d in data]))
    items = {
        item_name: list(set([d[key] for d in data if d[group_by] == item_name]))
        for item_name in item_names
    }
    for item, values in items.items():
        st.markdown(f"{item}: {values}")


def getRange(metric, data, padding_multipliers):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * padding_multipliers[0],
        maxV + padding * padding_multipliers[1],
    ]
    return rangeV


def drawChart(st, data: list[object], metric):
    unit = metricUnitMap.get(metric, "")
    x = "filter_rate"
    xrange = getRange(x, data, [0.05, 0.1])

    y = metric
    yrange = getRange(y, data, [0.2, 0.1])

    data.sort(key=lambda a: a[x])

    line_group = "db_name"
    color = "db"
    color_discrete_map = COLOR_MAP

    color_count = len(set([d[color] for d in data]))
    if color_count == 1:
        color = line_group
        line_group = None
        color_discrete_map = None

    fig = px.line(
        data,
        x=x,
        y=y,
        color=color,
        line_group=line_group,
        color_discrete_map=color_discrete_map,
        text=metric,
        markers=True,
    )
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)
    fig.update_traces(textposition="bottom right", texttemplate="%{y:,.4~r}" + unit)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
