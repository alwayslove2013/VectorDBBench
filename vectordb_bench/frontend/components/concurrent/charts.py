

from vectordb_bench.backend.cases import Case
from vectordb_bench.frontend.components.check_results.expanderStyle import initMainExpanderStyle
import plotly.express as px

from vectordb_bench.frontend.const.styles import COLOR_MAP


def drawChartsByCase(allData, cases: list[Case], st):
    initMainExpanderStyle(st)
    for case in cases:
        chartContainer = st.expander(case.name, True)
        caseDataList = [
            data for data in allData if data["case_name"] == case.name]
        data = [{
            "conc_num": caseData["conc_num_list"][i],
            "qps": caseData["conc_qps_list"][i],
            "latency_p99": caseData["conc_latency_p99_list"][i] * 1000,
            "db_name": caseData["db_name"],
            "db": caseData["db"]

        } for caseData in caseDataList for i in range(len(caseData["conc_num_list"]))]
        if len(data) == 0:
            continue
        drawChart(data, chartContainer)

        # errorDBs = failedTasks[case.name]
        # showFailedDBs(chartContainer, errorDBs)


def getRange(metric, data, padding_multipliers):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * padding_multipliers[0],
        maxV + padding * padding_multipliers[1],
    ]
    return rangeV


def drawChart(data, st):
    x = "latency_p99"
    xrange = getRange(x, data, [0.05, 0.1])

    y = "qps"
    yrange = getRange(y, data, [0.2, 0.1])

    # if len(selectedDbLabels) == 1:
    #     color = "type_and_clause_num"
    #     color_discrete_map = None
    # else:
    #     color = "dbLabel"
    #     color_discrete_map = colorMap
    color = "db"
    line_group = "db_name"
    text = "conc_num"

    data.sort(key=lambda a: a["conc_num"])

    fig = px.line(
        data,
        x=x,
        y=y,
        color=color,
        color_discrete_map=COLOR_MAP,
        line_group=line_group,
        text=text,
        markers=True,
        # color_discrete_map=color_discrete_map,
        hover_data={
            "conc_num": True,
        },
        height=720,
    )
    fig.update_xaxes(range=xrange, title_text="Latency P99 (ms)")
    fig.update_yaxes(range=yrange, title_text="QPS")
    fig.update_traces(textposition="bottom right",
                      texttemplate="conc-%{text:,.4~r}")
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""
        ),
    )
    st.plotly_chart(fig, use_container_width=True,)
