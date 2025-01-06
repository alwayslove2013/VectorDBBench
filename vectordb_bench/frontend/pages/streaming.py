import logging
import streamlit as st
from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.filter import FilterType
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToResults,
    NavToRunTest,
)
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.components.streaming.charts import drawChartsByCase
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults
from vectordb_bench.frontend.components.streaming.data import DisplayedMetric
from vectordb_bench.frontend.config.styles import FAVICON
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import TestResult

log = logging.getLogger("vectordb_bench")


def main():
    # set page config
    st.set_page_config(
        page_title="VDBB Streaming Perf",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    allResults = benchMarkRunner.get_results()

    def check_conc_data(res: TestResult):
        case_results = res.results
        flag = False
        for case_result in case_results:
            if case_result.task_config.case_config.case.label == CaseLabel.Streaming:
                flag = True

        return flag

    checkedResults = [res for res in allResults if check_conc_data(res)]

    st.title("VDBBench - Streaming Performance")

    # results selector
    resultSelectorContainer = st.sidebar.container()
    shownData, _, showCaseNames = getshownData(
        resultSelectorContainer,
        checkedResults,
        filter_type=FilterType.NonFilter,
    )

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToResults(navContainer)

    # save or share
    resultesContainer = st.sidebar.container()
    getResults(resultesContainer, "vectordb_bench_streaming")

    # # main
    # latency_type = st.radio("Latency Type", options=["latency_p99", "latency_avg"])
    st.markdown(
        "Tests search performance with a **stable** and **fixed** insertion rate."
    )
    compared_with_optimized = st.toggle(
        "Compare with **optimezed** performance.",
        value=True,
        help="VectorDB is allowed to do **optimizations** after all insertions done and then test search performance.",
    )
    show_ndcg = st.toggle(
        "Show **NDCG** instead of Recall.",
        value=False,
        help="A more appropriate indicator to measure ANN search accuracy than Recall.",
    )
    need_adjust = st.toggle(
        "Adjust the NDCG/Recall value based on the search stage.",
        value=True,
        help="NDCG/Recall is calculated using the ground truth file of the **entire** database, **divided by the search stage** to simulate the actual value.",
    )
    x_use_actual_time = st.toggle(
        "Use actual time as X-axis instead of search stage.",
        value=False,
        help="Since vdbbench inserts may be faster than vetordb can process them, the time it actually reaches search_stage may have different delays.",
    )
    accuracy_metric = DisplayedMetric.recall
    if show_ndcg:
        if need_adjust:
            accuracy_metric = DisplayedMetric.adjusted_ndcg
        else:
            accuracy_metric = DisplayedMetric.ndcg
    else:
        if need_adjust:
            accuracy_metric = DisplayedMetric.adjusted_recall
    line_chart_displayed_y_metrics: list[tuple[DisplayedMetric, str]] = [
        (
            DisplayedMetric.qps,
            "max-qps of increasing **concurrency search** tests in each search stage.",
        ),
        (accuracy_metric, "calculated in each search_stage."),
        (
            DisplayedMetric.latency_p99,
            "serial lantency (p99) of **serial search** tests in each search stage.",
        ),
    ]
    line_chart_displayed_x_metric = DisplayedMetric.search_stage
    if x_use_actual_time:
        line_chart_displayed_x_metric = DisplayedMetric.search_time

    drawChartsByCase(
        st.container(),
        shownData,
        showCaseNames,
        with_last_optimized_data=compared_with_optimized,
        line_chart_displayed_x_metric=line_chart_displayed_x_metric,
        line_chart_displayed_y_metrics=line_chart_displayed_y_metrics,
    )

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
