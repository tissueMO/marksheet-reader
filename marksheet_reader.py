###############################################################################
#    マークシートを読み取り、CSVに集計結果を出力します。
###############################################################################
import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import pathlib
import json
from tqdm import tqdm
from configparser import ConfigParser
from typing import Any, Dict, List, Tuple

# 独自モジュール
from logger import Logger


# 設定ファイル読み込み
config = ConfigParser()
config.read("settings.conf", encoding="utf-8")


class MarksheetReader():
    """マークシートの読み込みを行うクラスです。
    """

    def __init__(self, threshold: float, verbose: bool):
        """コンストラクター

        Arguments:
            threshold {float} -- マーカー点の認識閾値
            verbose {bool} -- ログを詳細に出力するかどうか
        """
        self.logger = Logger("MarksheetReader")

        # 引数からオプションをセット
        self.marker_threshold = threshold
        self.verbose = verbose

        # マーカー画像をグレースケールで読み込む
        self.marker = cv2.imread("image/marker.jpg", 0)

        # マーカーのサイズを出力
        self.marker_src_width, self.marker_src_height = self.marker.shape[::-1]
        self.logger.log_debug(
            f"マーカー原寸サイズ :W={self.marker_src_width},H={self.marker_src_height}"
        )

        # 解像度に合わせてマーカーのサイズを変更
        self.marker = cv2.resize(
            self.marker,
            tuple(json.loads(config.get("marksheet", "marker_dest_size")))
        )
        self.logger.log_debug(f"マーカー認識サイズ: {marker.shape[::-1]}")

        # 各種設定値を読み込む
        self._load_settings()

    def _load_settings(self):
        """各種設定値を読み込んでメンバー変数に格納します。
        """
        # マーカー設定
        self.marker_dpi = config.getint("marker", "marker_dpi")
        self.scan_dpi = config.getint("marker", "scan_dpi")

        # マークシート設定
        self.n_page = config.getint("marksheet", "n_page")
        self.n_col = config.getint("marksheet", "n_col")
        self.n_row = config.getint("marksheet", "n_row")
        self.margin_top = json.loads("marksheet", "margin_top")
        self.margin_bottom = json.loads("marksheet", "margin_bottom")
        self.total_row = config.getint("marksheet", "total_row")
        self.cell_size = config.getint("marksheet", "cell_size")
        self.gray_threshold = config.getint("marksheet", "gray_threshold")
        self.result_threshold_minrate = config.getfloat(
            "marksheet", "result_threshold_minrate"
        )
        self.marker_dest_size = tuple(
            json.loads(config.get("marksheet", "marker_dest_size"))
        )
        self.offset_top = config.getint("marksheet", "offset_top")
        self.offset_left = config.getint("marksheet", "offset_left")
        self.blur_strength = tuple(
            json.loads(config.get("marksheet", "blur_strength"))
        )

        # 集計設定
        self.summary_dir = config.get("summarize", "summary_dir")
        self.p_question_indexes = json.loads(
            config.get("summarize", "p_question_indexes")
        )

    def load_marksheet(self, filename: str) -> np.ndarray:
        """マークシート画像を読み込み、認識可能な状態に整形します。
        読み込みに失敗した場合は None を返します。

        Arguments:
            filename {str} -- ファイル名
        Returns:
            np.ndarray -- 抽出したマークシート部分の画像
        """
        basename = os.path.basename(filename)

        # スキャン画像の取り込み
        filename = str(pathlib.Path(filename).relative_to(pathlib.Path.cwd()))
        image = cv2.imread(filename, 0)
        if image is None:
            return None

        # スキャン画像を二値化
        _, image = cv2.threshold(
            image,
            self.gray_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # スキャン画像の中からマーカーを抽出
        res = cv2.matchTemplate(image, self.marker, cv2.TM_CCOEFF_NORMED)

        # 類似度の閾値以上の座標を取り出す
        loc = np.where(res >= marker_threshold)
        if len(loc) == 0 or len(loc[0]) == 0 or len(loc[1]) == 0:
            if self.verbose:
                self.logger.log_error(f"マーカーの認識に失敗 :basename={basename}")
            return None

        # 認識領域を切り出し
        mark_area = {}
        mark_area["top_x"] = \
            min(loc[1] + config.getint("marksheet", "offset_left"))
        mark_area["top_y"] = \
            min(loc[0]) + self.marker_dest_size[1] + \
            config.getint("marksheet", "offset_top")
        mark_area["bottom_x"] = max(loc[1])
        mark_area["bottom_y"] = max(loc[0])

        image = image[
            mark_area["top_y"]: mark_area["bottom_y"],
            mark_area["top_x"]: mark_area["bottom_x"]
        ]

        # 列数、行数ベースでキリのいいサイズにリサイズ
        image = cv2.resize(
            image,
            (self.n_col * self.cell_size, self.total_row * self.cell_size)
        )

        # 画像に軽くブラーをかけて２値化し、白黒反転させる（塗りつぶした部分が白く浮き上がる）
        image = cv2.GaussianBlur(image, blur_strength, 0)
        res, image = cv2.threshold(
            image,
            self.gray_threshold,
            255,
            cv2.THRESH_BINARY
        )
        image = 255 - image

        return image

    def recognize_marksheet(self, image: np.ndarray, filename: str) \
            -> Tuple[int, Array]:
        """読み込まれたマークシートをもとに、塗りつぶされた項目の列番号を認識して配列で返します。
        ここに渡す画像は二値化されており、かつ１行と１列でサイズが等しいことが前提となります。

        Arguments:
            image {np.ndarray} -- 読み取り対象の画像
            filename {str} -- ファイル名
        Returns:
            Tuple[int, Array] --
                int -- ページ番号。読み取れなかった場合は 0 を返す
                Array -- 各設問に対する回答情報の配列を格納した配列
        """
        basename = os.path.basename(filename)
        page_number = 0
        results = []

        # 行ごとに走査する
        for row in range(self.total_row - self.margin_bottom):
            if 0 < row and row < self.margin_top:
                # ページ番号行を除く、設問対象にしない余白行
                continue
            if 0 < row and row not in self.p_question_indexes[page_number - 1]:
                # 設問ではない行
                continue

            # 処理する行を切り出して 0-1 標準化
            row_image = image[
                row * self.cell_size: (row + 1) * self.cell_size, ]
            row_image = row_image / 255.0
            area_sum = []    # ここに合計値を入れる

            # 列ごとに走査する
            for col in range(self.n_col):
                # 各セルの領域について、画像の合計値（＝白い部分の面積）を求める
                cell_image = row_image[
                    :,
                    col * self.cell_size: (col + 1) * self.cell_size
                ]
                area_sum.append(np.sum(cell_image))

            # 各セルの合計値を上限値（＝全部塗りつぶしたときの理論値）に対する割合にする
            area_sum = np.ndarray(area_sum) / (self.cell_size * self.cell_size)
            max = np.max(area_sum)

            if max < self.result_threshold_minrate:
                # 最大値が閾値を下回っているときは空欄と判断
                answer_list = np.asarray([])
            else:
                # 暫定回答を出す
                result = area_sum > np.max(
                    np.asarray([max * 0.5, self.result_threshold_minrate])
                )
                result = np.asarray([1 if x else 0 for x in result])
                answer_list = self.get_answer(result)

            if row == 0:
                # ページ番号 (1 origin) として取り出す
                page_number_list = answer_list
                if page_number_list.shape[0] == 1:
                    page_number = page_number_list[0]

                if page_number == 0:
                    # ページ番号が不明だと設問構成も不明なので中断する
                    self.logger.log_error("ページ番号不明 :basename=" + basename)
                    return 0, None
            else:
                if answer_list.shape[0] > 0:
                    # 回答として取り出す
                    results.append(result)
                else:
                    # 未回答
                    results.append([])

        return page_number, results

    def get_answer(result):
        """塗りつぶしのデータから、回答を取り出します。

        Arguments:
            results {Array} -- 各設問に対する塗りつぶしの有無
        Returns:
            Array -- 回答番号 (1 origin)
        """
        data = np.where(result == 1)[0] + 1
        data = data.astype(np.uint8)
        return data
