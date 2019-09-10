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


if __name__ == "__main__":
    logger = Logger("__main__")
    reader = MarksheetReader(COMMANDLINE_OPTIONS.threshold)

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

    # 集計データのテーブル
    answers = []
    answer_tables = []
    data_sums = []
    n_page = len(reader.p_question_indices)

    for i in range(n_page):
        # このページの設問数を取得
        n_question = len(reader.p_question_indices[i])

        aggregates_columns_sum1 = {
            "Q-No.": [
                ("Q-" + str(row + 1))
                for row in range(n_question)
            ]
        }
        aggregates_columns_sum2 = {
            ("Ans-" + str(col + 1)): [
                0 for row in range(n_question)
            ]
            for col in range(reader.n_col)
        }
        aggregates_columns_sum = {
            **aggregates_columns_sum1,
            **aggregates_columns_sum2
        }
        data_sum = pd.DataFrame(aggregates_columns_sum)
        data_sum = data_sum.ix[
            :,
            [item[0] for item in aggregates_columns_sum1.items()] +
            [item[0] for item in aggregates_columns_sum2.items()]
        ]
        data_sums.append(data_sum)

        # 個別の回答一覧
        aggregates_columns_person0 = {
            "ファイル名": [],
            "ページ番号": [],
        }
        aggregates_columns_person1 = {
            "Q-No.": []
        }
        aggregates_columns_person2 = {
            ("Ans-" + str(col + 1)): [] for col in range(reader.n_col)
        }
        aggregates_columns_person = {
            **aggregates_columns_person0,
            **aggregates_columns_person1,
            **aggregates_columns_person2
        }
        answer_table = pd.DataFrame(aggregates_columns_person)
        answer_table = answer_table.ix[
            :,
            (
                [item[0] for item in aggregates_columns_person0.items()] +
                [item[0] for item in aggregates_columns_person1.items()] +
                [item[0] for item in aggregates_columns_person2.items()]
            )
        ]
        answer_tables.append(answer_table)

        # 個別の回答テキスト版
        answers.append([])

    # 複数回答の一覧
    multi_ans = pd.DataFrame({
        "ファイル名": [],
        "ページ番号": [],
        "設問番号": [],
        "答え？": [],
    })
    multi_ans = multi_ans.ix[:, ["ファイル名", "ページ番号", "設問番号", "答え？"]]

    # 未回答の一覧
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

    # マークシートのスキャン画像を逐一読み取って集計
    files = os.listdir(COMMANDLINE_OPTIONS.imgdir)
    logger.log_info("マークシート読み取り開始...")
    for file in tqdm(files):
        filename = os.path.join(COMMANDLINE_OPTIONS.imgdir, file)

        # 現在のファイルに対して回答チェック
        logger.log_debug(filename)

        image = reader.load_marksheet(filename)
        if image is None:
            # 認識エラー: 歪んでいるなどにより、マーカーを認識できなかった
            no_recognize = no_recognize.append(
                pd.Series(
                    [
                        file
                    ],
                    index=no_recognize.columns
                ),
                ignore_index=True
            )
            continue
        else:
            # マーク読み取り実行
            page_number, results = reader.recognize_marksheet(image, filename)

            if page_number == 0:
                # ページ番号が無効
                no_recognize = no_recognize.append(
                    pd.Series(
                        [
                            file
                        ],
                        index=no_recognize.columns
                    ),
                    ignore_index=True
                )
                continue

            answers[page_number - 1].append(os.path.basename(filename))
            answers[page_number - 1].append("")

        # 読み取り結果を集計
        for row, result in enumerate(results):
            data = reader.get_answer(result)

            row_data = [
                file,
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
                            file,
                            page_number,
                            int(row + 1),
                            str(data),
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
                            file,
                            page_number,
                            int(row + 1),
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
                f.write(line)
                f.write("\n")

    logger.log_info(f"集計結果を {reader.summary_dir} 以下 に書き出しました")
