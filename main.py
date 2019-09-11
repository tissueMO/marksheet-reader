# coding: utf-8
###############################################################################
#    マークシートを読み取り、CSVに集計結果を出力します。
###############################################################################
import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import argparse
import pathlib
import json
from tqdm import tqdm
from configparser import ConfigParser
from typing import Any, Dict, List, Tuple, List

# 独自モジュール
from marksheet_reader import MarksheetReader
from logger import Logger


# コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--imgdir",
    type=str,
    default="./sample",
    help="スキャンした画像のあるディレクトリーを指定して下さい。"
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="このオプションが指定された場合は、個々の読み取り結果を出力します。"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="マーカー点の認識閾値を指定して下さい。デフォルト値は 0.5 です。"
)
COMMANDLINE_OPTIONS = parser.parse_args()


def init_answer_columns(reader: MarksheetReader) -> Tuple[
        int, List, List, List]:
    """集計用データ列を初期化します。

    Arguments:
        reader {MarksheetReader} -- マークシートリーダーオブジェクト

    Returns:
        Tuple[int, List, List, List] --
            int -- 設定値から取得したフォームのページ数
            List -- 個別回答
            List -- ページごと回答者個別の集計テーブルのリスト
            List -- ページごとの集計テーブルのリスト
    """
    n_page = len(reader.p_question_indices)
    answers = []
    answer_tables = []
    data_sums = []

    for i in range(n_page):
        # このページの設問数を取得
        n_question = len(reader.p_question_indices[i])

        # 集計列を生成
        aggregates_columns_sum_question = {
            "Q-No.": [
                ("Q-" + str(row + 1))
                for row in range(n_question)
            ]
        }
        aggregates_columns_sum_answer = {
            ("Ans-" + str(col + 1)): [
                0 for row in range(n_question)
            ]
            for col in range(reader.n_col)
        }
        aggregates_columns_sum = {
            **aggregates_columns_sum_question,
            **aggregates_columns_sum_answer
        }
        data_sum = pd.DataFrame(aggregates_columns_sum)
        data_sum = data_sum.ix[
            :,
            [item[0] for item in aggregates_columns_sum_question.items()] +
            [item[0] for item in aggregates_columns_sum_answer.items()]
        ]
        data_sums.append(data_sum)

        # 個別の回答一覧
        aggregates_columns_person_basic = {
            "ファイル名": [],
            "ページ番号": [],
            "Q-No.": [],
        }
        aggregates_columns_person_answer = {
            ("Ans-" + str(col + 1)): []
            for col in range(reader.n_col)
        }
        aggregates_columns_person = {
            **aggregates_columns_person_basic,
            **aggregates_columns_person_answer
        }
        answer_table = pd.DataFrame(aggregates_columns_person)
        answer_table = answer_table.ix[
            :,
            (
                [item[0] for item in aggregates_columns_person_basic.items()] +
                [item[0] for item in aggregates_columns_person_answer.items()]
            )
        ]
        answer_tables.append(answer_table)

        # 個別の回答テキスト版
        answers.append([])

    return n_page, answers, answer_tables, data_sums


def init_warning_results() -> Tuple[List, List, List]:
    """要注意結果のデータフレームを初期化します。

    Returns:
        Tuple[List, List, List] --
            List -- 複数回答
            List -- 無回答
            List -- 読み取りエラー
    """
    # 複数回答の一覧
    multi_ans = pd.DataFrame({
        "ファイル名": [],
        "ページ番号": [],
        "設問番号": [],
        "答え？": [],
    })
    multi_ans = multi_ans.ix[:, ["ファイル名", "ページ番号", "設問番号", "答え？"]]

    # 無回答の一覧
    no_ans = pd.DataFrame({
        "ファイル名": [],
        "ページ番号": [],
        "設問番号": [],
    })
    no_ans = no_ans.ix[:, ["ファイル名", "ページ番号", "設問番号"]]

    # 読み取りエラーの一覧
    no_recognize = pd.DataFrame({
        "ファイル名": [],
    })
    no_recognize = no_recognize.ix[:, ["ファイル名", ]]

    return multi_ans, no_ans, no_recognize


def process_summarize(reader: MarksheetReader, file_name: str, answers: List,
                      data_sums: List, multi_ans: List, no_ans: List,
                      no_recognize: List, answer_tables: List) -> Tuple[
                          List, List, List]:
    """与えられた画像ファイルを読み込み、結果を格納します。

    Arguments:
        reader {MarksheetReader} -- マークシートリーダーオブジェクト
        file_name {str} -- 読み込み対象のファイル名（基準画像ディレクトリーまでの文字列を除いたもの）
        data_sums {List} -- ページごと設問ごとの集計結果
        multi_ans {List} -- 複数回答のリスト
        no_ans {List} -- 無回答のリスト
        no_recognize {List} -- 認識できなかったファイルのリスト
        answer_tables {List} -- 個人単位での読み取り結果のリスト

    Returns:
        Tuple[List, List, List] --
            List -- 更新後の multi_ans
            List -- 更新後の no_ans
            List -- 更新後の no_recognize
    """
    logger = Logger("process_summarize")
    file_path = os.path.join(COMMANDLINE_OPTIONS.imgdir, file_name)

    # 現在のファイルに対して回答チェック
    logger.log_debug(file_path)

    image = reader.load_marksheet(file_path)
    if image is None:
        # 認識エラー: 歪んでいるなどにより、マーカーを認識できなかった
        no_recognize = no_recognize.append(
            pd.Series([file_name], index=no_recognize.columns),
            ignore_index=True
        )
        return multi_ans, no_ans, no_recognize

    # マーク読み取り実行
    page_number, results = reader.recognize_marksheet(image, file_path)

    if page_number == 0:
        # ページ番号が無効
        no_recognize = no_recognize.append(
            pd.Series([file_name], index=no_recognize.columns),
            ignore_index=True
        )
        return multi_ans, no_ans, no_recognize

    answers[page_number - 1].append(os.path.basename(file_path))
    answers[page_number - 1].append("")

    # 読み取り結果を集計
    for row, result in enumerate(results):
        data = reader.get_answer(result)

        row_data = [
            file_name,
            page_number,
            row + 1,
        ]
        for i in range(reader.n_col):
            exists = ((i + 1) in data)
            row_data.append(exists)

        answer_table_row = pd.Series(
            row_data,
            answer_tables[page_number - 1].columns
        )

        if len(data) == 1:
            # 単一回答
            line = "Q-%02d. " % (row + 1) + str(data[0])
            if COMMANDLINE_OPTIONS.verbose:
                logger.log_debug(line)

            data_sums[page_number - 1].iat[row, data[0]] = str(
                int(data_sums[page_number - 1].iat[row, data[0]]) + 1
            )
            answers[page_number - 1].append(line)

        elif len(data) > 1:
            # 複数回答
            line = "Q-%02d. " % (row + 1) + str(data) + "  # 複数回答 #"
            logger.log_warn(line)

            multi_ans = multi_ans.append(
                pd.Series(
                    [
                        file_name,
                        f"{page_number}",
                        f"{row + 1}",
                        f"{data}",
                    ],
                    index=multi_ans.columns
                ),
                ignore_index=True
            )
            answers[page_number - 1].append(line)

        else:
            # 無回答
            line = "Q-%02d. " % (row + 1) + "** 未回答 **"
            logger.log_warn(line)

            no_ans = no_ans.append(
                pd.Series(
                    [
                        file_name,
                        f"{page_number}",
                        f"{row + 1}",
                    ],
                    index=no_ans.columns
                ),
                ignore_index=True
            )
            answers[page_number - 1].append(line)

        answer_tables[page_number - 1] = \
            answer_tables[page_number - 1].append(
                answer_table_row, ignore_index=True)

    answers[page_number - 1].append("\n--------------------------------\n")
    return multi_ans, no_ans, no_recognize


def print_summary(
            reader: MarksheetReader, n_page: int, data_sums: List,
            multi_ans: List, no_ans: List, no_recognize: List,
            answer_tables: List):
    """マークシートの集計結果を標準出力・ファイルに出力します。

    Arguments:
        reader {MarksheetReader} -- マークシートリーダーオブジェクト
        n_page {int} -- フォームのページ数
        data_sums {List} -- ページごと設問ごとの集計結果
        multi_ans {List} -- 複数回答のリスト
        no_ans {List} -- 無回答のリスト
        no_recognize {List} -- 認識できなかったファイルのリスト
        answer_tables {List} -- 個人単位での読み取り結果のリスト
    """
    logger = Logger("print_summary")

    logger.log_debug("\n◆集計結果\n")
    for i in range(n_page):
        logger.log_debug(f"Page:{i + 1}\n{data_sums[i]}\n")
    logger.log_debug(f"◆複数回答\n{multi_ans}\n")
    logger.log_debug(f"◆無回答\n{no_ans}\n")
    logger.log_debug(f"◆認識エラー\n{no_recognize}\n\n")

    # 集計データをCSVに出力
    if not os.path.isdir(reader.summary_dir):
        os.mkdir(reader.summary_dir)
    for i in range(n_page):
        data_sums[i].to_csv(
            os.path.join(
                reader.summary_dir, "aggregates-p" + str(i + 1) + ".csv"
            ),
            index=False,
            encoding="sjis"
        )
        answer_tables[i].to_csv(
            os.path.join(
                reader.summary_dir, "answers-p" + str(i + 1) + ".csv"
            ),
            index=False,
            encoding="sjis"
        )

    # 要注意CSVを出力
    multi_ans.to_csv(
        os.path.join(reader.summary_dir, "multiple_answers.csv"),
        index=False,
        encoding="sjis"
    )
    no_ans.to_csv(
        os.path.join(reader.summary_dir, "nothing_answers.csv"),
        index=False,
        encoding="sjis"
    )
    no_recognize.to_csv(
        os.path.join(reader.summary_dir, "no_recognized.csv"),
        index=False,
        encoding="sjis"
    )

    # ページごと、ファイルごとの個別回答情報を書き出し
    for i, answer_page in enumerate(answers):
        with open(
                os.path.join(
                    reader.summary_dir,
                    "answers-p" + str(i + 1) + ".txt"), "w"
        ) as f:
            for line in answer_page:
                f.write(f"{line}\n")

    logger.log_info(f"集計結果を {reader.summary_dir} 以下 に書き出しました")


"""メインルーチン
"""
if __name__ == "__main__":
    logger = Logger("__main__")
    reader = MarksheetReader(COMMANDLINE_OPTIONS.threshold, COMMANDLINE_OPTIONS.verbose)

    # コマンドライン引数チェック
    if COMMANDLINE_OPTIONS.imgdir is None \
            or not os.path.isdir(COMMANDLINE_OPTIONS.imgdir):
        logger.log_error("--imgdir [必須] 取り込む画像が含まれるディレクトリーを指定して下さい")
        sys.exit()
    logger.log_info(
        f"コマンドライン引数" +
        f" :imgdir={COMMANDLINE_OPTIONS.imgdir}" +
        f" :verbose={COMMANDLINE_OPTIONS.verbose}" +
        f" :threshold={COMMANDLINE_OPTIONS.threshold}"
    )

    # 各種テーブル初期化
    n_page, answers, answer_tables, data_sums = init_answer_columns(reader)
    multi_ans, no_ans, no_recognize = init_warning_results()

    # マークシートのスキャン画像を逐一読み取って集計
    files = os.listdir(COMMANDLINE_OPTIONS.imgdir)
    logger.log_info("マークシート読み取り開始...")
    for file in tqdm(files):
        multi_ans, no_ans, no_recognize = process_summarize(
            reader, file, answers,
            data_sums, multi_ans, no_ans,
            no_recognize, answer_tables
        )

    # 結果を出力
    print_summary(
        reader, n_page, data_sums, multi_ans, no_ans,
        no_recognize, answer_tables
    )
