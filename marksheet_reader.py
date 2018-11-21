# coding: UTF-8
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
from tqdm import tqdm


# 読み取り設定
import settings


##### コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", type = str, default = None, help = "スキャンした画像のあるディレクトリーです。")
parser.add_argument("--verbose", action = "store_true", help = "個々の読み取り結果を表示します。")
parser.add_argument("--threshold", type = float, default = 0.5, help = "マーカー点の認識閾値です。")
FLAGS = parser.parse_args()


##### マーカー画像読み込み
# マーカー画像をグレースケールで読み込む
marker = cv2.imread("image/marker.jpg", 0)
marker_threshold = FLAGS.threshold   # マーカー点の認識閾値

# マーカーのサイズを保管
marker_src_width, marker_src_height = marker.shape[::-1]
print("マーカー原寸サイズ:", "W =", marker_src_width, ", H =", marker_src_height)

# 解像度に合わせてマーカーのサイズを変更
# settings.marker_dest_width, settings.marker_dest_height = (int(marker_src_height * settings.scan_dpi / settings.marker_dpi), int(marker_src_width * settings.scan_dpi / settings.marker_dpi))
marker = cv2.resize(marker, (settings.marker_dest_width, settings.marker_dest_height))
print("マーカー認識サイズ:", marker.shape[::-1])



def loadMarkSheet(filename):
	"""マークシートを読み込み、認識できる状態に整形します。
	認識に失敗した場合は None を返します。
	Arguments:
		filename {string} -- ファイル名
	Returns:
		Image -- 抽出したマークシート部分の画像
	"""
	basename = os.path.basename(filename)

	# スキャン画像の取り込み
	filename = str(pathlib.Path(filename).relative_to(pathlib.Path.cwd()))
	image = cv2.imread(filename, 0)
	if image is None:
		return None

	# スキャン画像を二値化
	_, image = cv2.threshold(image, settings.gray_threshold, 255, cv2.THRESH_BINARY)
	# cv2.imwrite(os.path.join(basename + "-scan_bin.jpg"), image)

	# スキャン画像の中からマーカーを抽出
	res = cv2.matchTemplate(image, marker, cv2.TM_CCOEFF_NORMED)
	# print(res)

	# 類似度の閾値以上の座標を取り出す
	loc = np.where(res >= marker_threshold)
	if len(loc) == 0 or len(loc[0]) == 0 or len(loc[1]) == 0:
		if FLAGS.verbose:
			print("マーカーの認識に失敗")
		return None

	# 認識領域を切り出し
	mark_area = {}
	mark_area["top_x"] = min(loc[1] + settings.offset_left)
	mark_area["top_y"] = min(loc[0]) + settings.marker_dest_height + settings.offset_top
	mark_area["bottom_x"] = max(loc[1])
	mark_area["bottom_y"] = max(loc[0])
	# print(mark_area)

	image = image[mark_area["top_y"] : mark_area["bottom_y"], mark_area["top_x"] : mark_area["bottom_x"]]

	# 切り出した領域を検証用に書き出し
	# cv2.imwrite(os.path.join(basename + "-scan_cropped.jpg"), image)

	# 列数、行数ベースでキリのいいサイズにリサイズ
	image = cv2.resize(image, (settings.n_col * settings.cell_size, settings.total_row * settings.cell_size))

	# 画像に軽くブラーをかけて２値化し、白黒反転させる（塗りつぶした部分が白く浮き上がる）
	image = cv2.GaussianBlur(image, (5, 5), 0)
	# res, image = cv2.threshold(image, settings.gray_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	res, image = cv2.threshold(image, settings.gray_threshold, 255, cv2.THRESH_BINARY)
	image = 255 - image

	# 認識可能な状態の画像を検証用に書き出し
	# cv2.imwrite(os.path.join(basename + "-scan_inverted.jpg"), image)

	return image



def recognizeMarkSheet(image, filename):
	"""読み込まれたマークシートをもとに、塗りつぶされた項目の列番号を認識して配列で返します。
	ここに渡す画像は二値化されており、かつ１行と１列でサイズが等しいことが前提となります。
	Arguments:
		image {Image} -- 読み取り対象の画像
		filename {string} -- ファイル名
	Returns:
		int -- ページ番号。読み取れなかった場合は 0 となります
		Array -- 各設問に対する回答情報の配列を格納した配列
	"""
	basename = os.path.basename(filename)

	page_number = 0
	results = []

	# 塗りつぶしの限界値
	# area_width, area_height = (settings.cell_size, settings.cell_size)
	# print("塗りつぶし面積最大値:", area_width * area_height)

	# 行ごとに走査する
	for row in range(settings.total_row - settings.margin_bottom):
		if 0 < row and row < settings.margin_top:
			# ページ番号行を除く、設問対象にしない余白行
			continue
		if 0 < row and not row in settings.p_question_indexes[page_number - 1]:
			# 設問ではない行
			continue

		# 処理する行を切り出して 0-1 標準化
		row_image = image[row * settings.cell_size : (row + 1) * settings.cell_size, ]
		# cv2.imwrite(os.path.join(basename + "-row" + str(row) + ".jpg"), row_image)
		row_image = row_image / 255.0
		area_sum = []	# ここに合計値を入れる

		# 列ごとに走査する
		for col in range(settings.n_col):
			# 各セルの領域について、画像の合計値（＝白い部分の面積）を求める
			cell_image = row_image[:, col * settings.cell_size : (col + 1) * settings.cell_size]
			area_sum.append(np.sum(cell_image))

		# 各セルの合計値を上限値（＝全部塗りつぶしたときの理論値）で割って割合にする
		area_sum = np.array(area_sum) / (settings.cell_size * settings.cell_size)
		max = np.max(area_sum)

		if max < settings.result_threshold_minrate:
			# 最大値が閾値を下回っているときは空欄と判断
			answer_list = np.asarray([])
		else:
			# 暫定回答を出す
			result = area_sum > np.max(np.asarray([max * 0.5, settings.result_threshold_minrate]))
			result = np.asarray([1 if x == True else 0 for x in result])
			answer_list = getAnswer(result)

		if row == 0:
			# ページ番号 (1 origin) として取り出す
			page_number_list = answer_list
			if page_number_list.shape[0] == 1:
				page_number = page_number_list[0]

			if page_number == 0:
				# ページ番号が不明だと設問構成も不明なので中断する
				print("ページ番号不明:", filename)
				return 0, None
		else:
			# 回答として取り出す
			if answer_list.shape[0] > 0:
				results.append(result)
			else:
				# 未回答
				results.append([])

	return page_number, results



def getAnswer(result):
	"""塗りつぶしのデータから、回答を取り出します。
	Arguments:
		results {Array} -- 各設問に対する塗りつぶしの有無
	Returns:
		Array -- 回答番号 (1 origin)
	"""
	data = np.where(result == 1)[0] + 1
	data = data.astype(np.uint8)
	return data



##### メインルーチン
if __name__ == '__main__':
	# コマンドライン引数チェック
	if FLAGS.imgdir is None or os.path.isdir(FLAGS.imgdir) == False:
		print("--imgdir: 取り込む画像が含まれるディレクトリーを指定してください")
		sys.exit()

	# 集計データのテーブル
	answers = []
	answer_tables = []
	data_sums = []
	for i in range(settings.n_page):
		# このページの設問数を取得
		n_question = len(settings.p_question_indexes[i])

		aggregates_columns_sum1 = { "Q-No.": [("Q-" + str(row + 1)) for row in range(n_question)] }
		aggregates_columns_sum2 = { ("Ans-" + str(col + 1)): [0 for row in range(n_question)] for col in range(settings.n_col) }
		aggregates_columns_sum = {**aggregates_columns_sum1, **aggregates_columns_sum2}
		data_sum = pd.DataFrame(aggregates_columns_sum)
		data_sum = data_sum.ix[:, [item[0] for item in aggregates_columns_sum1.items()] + [item[0] for item in aggregates_columns_sum2.items()]]
		data_sums.append(data_sum)

		# 個別の回答一覧
		aggregates_columns_person0 = {
			"ファイル名": [],
			"ページ番号": [],
		}
		aggregates_columns_person1 = { "Q-No.": [] }
		aggregates_columns_person2 = { ("Ans-" + str(col + 1)): [] for col in range(settings.n_col) }
		aggregates_columns_person = {**aggregates_columns_person0, **aggregates_columns_person1, **aggregates_columns_person2}
		answer_table = pd.DataFrame(aggregates_columns_person)
		answer_table = answer_table.ix[:, (
			[item[0] for item in aggregates_columns_person0.items()]
			+ [item[0] for item in aggregates_columns_person1.items()]
			+ [item[0] for item in aggregates_columns_person2.items()])
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
	no_recognize = no_recognize.ix[:, ["ファイル名",]]

	# マークシートのスキャン画像を逐一読み取って集計
	files = os.listdir(FLAGS.imgdir)
	print("マークシート読み取り開始...")
	for file in tqdm(files):
		filename = os.path.join(FLAGS.imgdir, file)

		# 現在のファイルに対して回答チェック
		print(filename)

		image = loadMarkSheet(filename)
		if image is None:
			# 認識エラー: 歪んでいるなどにより、マーカーを認識できなかった
			no_recognize = no_recognize.append(
				pd.Series(
					[
						file
					],
					index = no_recognize.columns
				),
				ignore_index = True
			)
			continue
		else:
			# マーク読み取り実行
			page_number, results = recognizeMarkSheet(image, filename)

			if page_number == 0:
				# ページ番号が無効
				no_recognize = no_recognize.append(
					pd.Series(
						[
							file
						],
						index = no_recognize.columns
					),
					ignore_index = True
				)
				continue

			answers[page_number - 1].append(os.path.basename(filename))
			answers[page_number - 1].append("")

		# 読み取り結果を集計
		for row, result in enumerate(results):
			data = getAnswer(result)

			row_data = [
				file,
				page_number,
				row + 1,
			]
			for i in range(settings.n_col):
				exists = (i in data)
				row_data.append(exists)

			answer_table_row = pd.Series(row_data, answer_tables[page_number - 1].columns)

			if len(data) == 1:
				# 単一回答
				line = "Q-%02d. " % (row + 1) + str(data[0])
				if FLAGS.verbose:
					print(line)

				data_sums[page_number - 1].iat[row, data[0]] = str(int(data_sums[page_number - 1].iat[row, data[0]]) + 1)
				answers[page_number - 1].append(line)

			elif len(data) > 1:
				# 複数回答
				line = "Q-%02d. " % (row + 1) + str(data) + "  # 複数回答 #"
				print(line)

				multi_ans = multi_ans.append(
					pd.Series(
						[
							file,
							page_number,
							int(row + 1),
							str(data),
						],
						index = multi_ans.columns
					),
					ignore_index = True
				)
				answers[page_number - 1].append(line)

			else:
				# 無回答
				line = "Q-%02d. " % (row + 1) + "** 未回答 **"
				print(line)

				no_ans = no_ans.append(
					pd.Series(
						[
							file,
							page_number,
							int(row + 1),
						],
						index = no_ans.columns
					),
					ignore_index = True
				)
				answers[page_number - 1].append(line)

			answer_tables[page_number - 1] = answer_tables[page_number - 1].append(answer_table_row, ignore_index = True)

		answers[page_number - 1].append("\n--------------------------------------------------------\n")

	print()
	print("◆集計結果\n")
	for i in range(settings.n_page):
		print("Page:", (i + 1), "\n", data_sums[i], "\n")
	print("◆複数回答\n", multi_ans, "\n")
	print("◆無回答\n", no_ans, "\n")
	print("◆認識エラー\n", no_recognize, "\n")
	print()


	# 集計データをCSVに出力
	if os.path.isdir(settings.summary_dir) == False:
		os.mkdir(settings.summary_dir)
	for i in range(settings.n_page):
		data_sums[i].to_csv(
			os.path.join(settings.summary_dir, "aggregates-p" + str(i + 1) + ".csv"),
			index = False,
			encoding = "sjis"
		)
		answer_tables[i].to_csv(
			os.path.join(settings.summary_dir, "answers-p" + str(i + 1) + ".csv"),
			index = False,
			encoding = "sjis"
		)
	multi_ans.to_csv(
		os.path.join(settings.summary_dir, "multiple_answers.csv"),
		index = False,
		encoding = "sjis"
	)
	no_ans.to_csv(
		os.path.join(settings.summary_dir, "nothing_answers.csv"),
		index = False,
		encoding = "sjis"
	)
	no_recognize.to_csv(
		os.path.join(settings.summary_dir, "no_recognized.csv"),
		index = False,
		encoding = "sjis"
	)

	# ページごと、ファイルごとの個別回答情報を書き出し
	for i, answer_page in enumerate(answers):
		with open(os.path.join(settings.summary_dir, "answers-p" + str(i + 1) + ".txt"), "w") as f:
			for line in answer_page:
				f.write(line)
				f.write("\n")

	print("集計結果を {", settings.summary_dir, "以下 } に書き出しました")
