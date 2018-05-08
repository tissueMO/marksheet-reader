###############################################################################
#    マークシートを読み取り、CSVに集計結果を出力します
###############################################################################
import numpy as np
import pandas as pd
import cv2
import os
import sys
import math
import argparse


##### コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument("--imgdir", type = str, default = None, help = "スキャンした画像のあるディレクトリーです。")
parser.add_argument("--verbose", action = "store_true", help = "個々の読み取り結果を表示します。")
FLAGS = parser.parse_args()


##### マーカー設定
marker_dpi = 112			# マーカーサイズ
scan_dpi = 400				# スキャン画像の解像度
marker_threshold = 0.5		# マーカー点の認識閾値

# マーカー画像をグレースケールで読み込む
marker = cv2.imread("marker.jpg", 0)

# マーカーのサイズを保管
marker_width, marker_height = (64, 64)
# marker_width, marker_height = marker.shape[::-1]
# print("W =", marker_width, ", H =", marker_height)

# 解像度に合わせてマーカーのサイズを変更
marker = cv2.resize(marker, (marker_width, marker_height))
# marker = cv2.resize(marker, (int(marker_height * scan_dpi / marker_dpi), int(marker_width * scan_dpi / marker_dpi)))
# print("マーカー認識サイズ:", marker.shape[::-1])


##### マークシート設定
n_col = 5							# マークシートの列数＝一行あたりのマーク数
n_row = 11							# マークシートの行数（≠ 項目数）
n_question = math.ceil(n_row / 2.0)	# マークシートの項目数（偶数行は空白にする＝行間を空けてマークしやすくする）
margin_top = 5						# 上部余白の行数
margin_bottom = 3					# 下部余白の行数
total_row = n_row + margin_top + margin_bottom		# 余白を含めた行数
size = 100							# １行１列あたりのサイズ
gray_threshold = 50					# 二値化の閾値
result_threshold_minrate = 0.1		# 塗りつぶしていると判断する最小割合の閾値
result_threshold_rate = 3.0			# 塗りつぶしていると決定する中央値からの倍率


# 集計設定
summary_dir = "summary"				# 書き出し先のディレクトリー名


##### マークシートを読み込み、認識できる状態に整形します。
# 認識に失敗した場合は None を返します。
def loadMarkSheet(filename):
	global marker, n_col, total_row, margin_top, margin_bottom, size, gray_threshold

	# スキャン画像の取り込み
	image = cv2.imread(filename, 0)

	# スキャン画像の中からマーカーを抽出
	res = cv2.matchTemplate(image, marker, cv2.TM_CCOEFF_NORMED)

	# 類似度の閾値以上の座標を取り出す
	loc = np.where(res >= marker_threshold)
	if len(loc) == 0 or len(loc[0]) == 0 or len(loc[1]) == 0:
		if FLAGS.verbose:
			print("マーカーの認識に失敗")
		return None

	# 認識領域を切り出し
	mark_area = {}
	mark_area["top_x"] = min(loc[1])
	mark_area["top_y"] = min(loc[0])
	mark_area["bottom_x"] = max(loc[1] + marker_width)
	mark_area["bottom_y"] = max(loc[0])

	image = image[mark_area["top_y"] : mark_area["bottom_y"], mark_area["top_x"] : mark_area["bottom_x"]]

	# 切り出した領域を検証用に書き出し
	# cv2.imwrite(os.path.join(path, "scan_cropped.jpg"), image)

	# 列数、行数ベースでキリのいいサイズにリサイズ
	image = cv2.resize(image, (n_col * size, total_row * size))

	# 画像に軽くブラーをかけて２値化し、白黒反転させる（塗りつぶした部分が白く浮き上がる）
	image = cv2.GaussianBlur(image, (5, 5), 0)
	res, image = cv2.threshold(image, gray_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	image = 255 - image

	# 認識可能な状態の画像を検証用に書き出し
	# cv2.imwrite(os.path.join(path, "scan_inverted.jpg"), image)

	return image


##### 読み込まれたマークシートをもとに、塗りつぶされた項目の列番号を認識して配列で返します。
# ここに渡す画像は二値化されており、かつ１行と１列でサイズが等しいことが前提となります。
def recognizeMarkSheet(image):
	global n_col, total_row, margin_top, margin_bottom, size

	results = []

	# 塗りつぶしの限界値
	# area_width, area_height = (size, size)
	# print("塗りつぶし面積最大値:", area_width * area_height)

	# 行ごとに走査する
	for row in range(margin_top, total_row - margin_bottom):
		if row % 2 == 0:
			# 偶数行は空白にする（行間を空けてマークしやすくする）ため飛ばす
			continue

		# 処理する行を切り出して 0-1 標準化
		row_image = image[row * size : (row + 1) * size, ] / 255.0
		area_sum = []	# ここに合計値を入れる

		# 列ごとに走査する
		for col in range(n_col):
			# 各セルの領域について、画像の合計値（＝白い部分の面積）を求める
			cell_image = row_image[:, col * size : (col + 1) * size]
			area_sum.append(np.sum(cell_image))
			# print(cell_image[int(size / 2), int(size / 2)])

		# 各セルの合計値を限界値で割って割合にする
		area_sum = np.array(area_sum) / (size * size)
		max = np.max(area_sum)
		med = np.median(area_sum)
		# print(max, area_sum)

		if max >= result_threshold_minrate:
			# 最大値が閾値を上回っており、かつ、塗りつぶされた部分の面積が中央値の３倍以上かどうかで結果を論理値判定する
			results.append(area_sum > med * result_threshold_rate)
		else:
			# 最大値が閾値を下回っている場合、無効票とする
			results.append([])

	return results


##### メインルーチン
if __name__ == '__main__':
	# コマンドライン引数チェック
	if FLAGS.imgdir is None or os.path.isdir(FLAGS.imgdir) == False:
		print("--imgdir: 取り込む画像が含まれるディレクトリーを指定してください")
		sys.exit()

	# 集計データのテーブル
	aggregates_columns1 = { "Q-No.": [("Q-" + str(row + 1)) for row in range(n_question)] }
	aggregates_columns2 = { ("Ans-" + str(col + 1)): [0 for row in range(n_question)] for col in range(n_col) }
	aggregates_columns = {**aggregates_columns1, **aggregates_columns2}
	data_sum = pd.DataFrame(aggregates_columns)
	data_sum = data_sum.ix[:, [item[0] for item in aggregates_columns1.items()] + [item[0] for item in aggregates_columns2.items()]]

	# 複数回答の一覧
	multi_ans = pd.DataFrame({
		"ファイル名": [],
		"設問番号": [],
		"答え？": [],
	})
	multi_ans = multi_ans.ix[:, ["ファイル名", "設問番号", "答え？"]]

	# 未回答の一覧
	no_ans = pd.DataFrame({
		"ファイル名": [],
		"設問番号": [],
	})
	no_ans = no_ans.ix[:, ["ファイル名", "設問番号"]]

	# 読み取りエラーの一覧
	no_recognize = pd.DataFrame({
		"ファイル名": [],
	})
	no_recognize = no_recognize.ix[:, ["ファイル名"]]

	# マークシートのスキャン画像を逐一読み取って集計
	files = os.listdir(FLAGS.imgdir)
	for file in files:
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
			results = recognizeMarkSheet(image)

		# 読み取り結果を集計
		for row, result in enumerate(results):
			data = np.where(result == True)[0] + 1

			if len(data) == 1:
				# 単一回答
				if FLAGS.verbose:
					print("Q-%02d. " % (row + 1) + str(data))

				data_sum.iat[row, int(data[0]) - 1] = data_sum.iat[row, int(data[0]) - 1] + 1

			elif len(data) > 1:
				# 複数回答
				if FLAGS.verbose:
					print("Q-%02d. " % (row + 1) + str(data) + "  # 複数回答 #")

				multi_ans = multi_ans.append(
					pd.Series(
						[
							file,
							row + 1,
							str(data),
						],
						index = multi_ans.columns
					),
					ignore_index = True
				)

			else:
				# 無回答
				if FLAGS.verbose:
					print("Q-%02d. " % (row + 1) + "** 未回答 **")

				no_ans = no_ans.append(
					pd.Series(
						[
							file,
							row + 1,
						],
						index = no_ans.columns
					),
					ignore_index = True
				)

	print()
	print("◆集計結果\n", data_sum, "\n")
	print("◆複数回答\n", multi_ans, "\n")
	print("◆無回答\n", no_ans, "\n")
	print("◆認識エラー\n", no_recognize, "\n")
	print()


	# 集計データをCSVに出力
	if os.path.isdir(summary_dir) == False:
		os.mkdir(summary_dir)
	data_sum.to_csv(
		os.path.join(summary_dir, "aggregates.csv"),
		index = False,
		encoding = "sjis"
	)
	multi_ans.to_csv(
		os.path.join(summary_dir, "multiple_answers.csv"),
		index = False,
		encoding = "sjis"
	)
	no_ans.to_csv(
		os.path.join(summary_dir, "nothing_answers.csv"),
		index = False,
		encoding = "sjis"
	)
	no_recognize.to_csv(
		os.path.join(summary_dir, "no_recognized.csv"),
		index = False,
		encoding = "sjis"
	)
	print("集計結果を {", summary_dir, "以下 } に書き出しました")
