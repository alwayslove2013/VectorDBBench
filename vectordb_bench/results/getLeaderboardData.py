from vectordb_bench import config
import ujson
import pathlib
import math
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.frontend.const.dbPrices import DB_DBLABEL_TO_PRICE
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import CaseResult, ResultLabel, TestResult

taskLabelToCode = {
    ResultLabel.FAILED: -1,
    ResultLabel.OUTOFRANGE: -2,
    ResultLabel.NORMAL: 1,
}


def main():
    allResults: list[TestResult] = benchMarkRunner.get_results()
    results: list[CaseResult] = []
    for result in allResults:
        if result.task_label == "standard":
            results += result.results

    if allResults is not None:
        data = [
            {
                "db": d.task_config.db.value,
                "db_label": d.task_config.db_config.db_label,
                "db_name": d.task_config.db_name,
                "case": d.task_config.case_config.case_id.case_name,
                "qps": d.metrics.qps,
                "latency": d.metrics.serial_latency_p99,
                "label": taskLabelToCode[d.label],
            }
            for d in results
            if d.task_config.case_config.case_id != CaseType.CapacityDim128
            and d.task_config.case_config.case_id != CaseType.CapacityDim960
        ]

        # compute qp$
        for d in data:
            db = d["db"]
            db_label = d["db_label"]
            qps = d["qps"]
            price = DB_DBLABEL_TO_PRICE.get(db, {}).get(db_label, 0)
            d["qp$"] = qps / price if price > 0 else 0.0

        # with open(pathlib.Path(config.RESULTS_LOCAL_DIR, "leaderboard.json"), "w") as f:
        #     ujson.dump(data, f)

        caseList = list(set([d["case"] for d in data]))
        dbNameList = list(set([d["db_name"] for d in data]))

        for metric in ["qps", "qp$"]:
            for case in caseList:
                caseData = [d for d in data if d["case"] == case]

                maxV = max([d[metric] for d in caseData])
                minV = min(
                    [d[metric] for d in caseData if d["label"] > 0 and d[metric] > 0]
                )

                for d in caseData:
                    v = d[metric] if d["label"] > 0 else (minV / 2)
                    d[f"{metric}-score"] = v / maxV * 100
                    assert d[f"{metric}-score"] <= 100, f"{d[metric]}, {maxV} >100????"

            dbScores = {dbName: {"count": 0, "scores": 1} for dbName in dbNameList}
            for d in data:
                if d[f"{metric}-score"] > 0:
                    dbScores[d["db_name"]]["scores"] *= d[f"{metric}-score"]
                    dbScores[d["db_name"]]["count"] += 1

            print("===>", metric)
            for dbName in dbNameList:
                if dbScores[dbName]["count"] > 0:
                    dbScores[dbName]["score"] = math.pow(
                        dbScores[dbName]["scores"], 1 / dbScores[dbName]["count"]
                    )

            dbScoreList = [
                (dbName, d.get("score", 0)) for dbName, d in dbScores.items()
            ]
            dbScoreList.sort(key=lambda x: x[1], reverse=True)
            for dbScore in dbScoreList:
                print(dbScore)

        for metric in ["latency"]:
            for case in caseList:
                caseData = [d for d in data if d["case"] == case]

                maxV = max([d[metric] for d in caseData])
                minV = min(
                    [d[metric] for d in caseData if d["label"] > 0 and d[metric] > 0]
                )

                for d in caseData:
                    v = d[metric] if d["label"] > 0 else (maxV * 2)
                    d[f"{metric}-score"] = (v + 10) / (minV + 10)

            dbScores = {dbName: {"count": 0, "scores": 1} for dbName in dbNameList}
            for d in data:
                if d[f"{metric}-score"] > 0:
                    dbScores[d["db_name"]]["scores"] *= d[f"{metric}-score"]
                    dbScores[d["db_name"]]["count"] += 1

            print("===>", metric)
            for dbName in dbNameList:
                if dbScores[dbName]["count"] > 0:
                    dbScores[dbName]["score"] = math.pow(
                        dbScores[dbName]["scores"], 1 / dbScores[dbName]["count"]
                    )

            dbScoreList = [
                # (dbName, d["score"], d["scores"], d["count"])
                (dbName, d.get("score", 0))
                for dbName, d in dbScores.items()
            ]
            dbScoreList.sort(key=lambda x: x[1])
            for dbScore in dbScoreList:
                print(dbScore)


if __name__ == "__main__":
    main()
