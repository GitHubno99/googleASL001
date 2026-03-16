from google.cloud import aiplatform
import json

import json
import ast
from scipy.optimize import minimize
from google.cloud import aiplatform

# プロジェクト情報
PROJECT_ID = "qwiklabs-asl-00-8534302ca246"
LOCATION = "us-central1"
ENDPOINT_ID_LINEAR = "1142366742736011264"
ENDPOINT_ID_NW = "199495939499491328"
ENDPOINT_ID_GAUSS = "809557767147749376"

# 初期化を確実に行う
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def _predict_vertex(endpoint_id: str, instances: list) -> str:
    """Vertex AI エンドポイントへの共通予測関数"""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    # 予測の実行
    response = endpoint.predict(instances=instances)
    return str(response.predictions)

def predict_linear_model(input_data=None, **kwargs) -> str:
    try:
        raw_data = input_data if input_data else kwargs
        if isinstance(raw_data, str):
            import ast, json
            try:
                raw_data = ast.literal_eval(raw_data)
            except:
                raw_data = json.loads(raw_data.replace("'", '"'))
        
        feature_keys = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
        feature_values = []
        missing_keys = [] # 不足している変数をメモするリスト
        
        for key in feature_keys:
            val = raw_data.get(key)
            
            # 値が存在しない（None）かチェック
            if val is None:
                missing_keys.append(key)
            else:
                if isinstance(val, list) and len(val) > 0:
                    feature_values.append(val[0])
                else:
                    feature_values.append(val)
        
        # 【追加した安全装置】不足しているキーがあれば、エラーメッセージとしてGeminiに返す
        if len(missing_keys) > 0:
            return f"予測を実行できません。以下の変数が不足しています: {missing_keys}。ユーザーにこれらの値を聞いてください。"
                
        # --- これ以降は元の予測実行コード ---
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID_LINEAR)
        response = endpoint.predict(instances=[feature_values])
        return f"線形モデルの計算結果: {response.predictions[0]}"
        
    except Exception as e:
        return f"ツール内部でデータ処理エラーが発生しました: {str(e)}"


def predict_nw_model(input_data=None, **kwargs) -> str:
    """
    ニューラルネットワーク（NW）モデルを使用して数値を算出します。
    複雑なパターン認識や、高度な非線形計算が必要な場合に使用します。
    """
    # 1. データの正規化
    raw_data = input_data if input_data else kwargs
    
    # LLMが気まぐれに「文字列」で送ってきた場合の救済措置
    if isinstance(raw_data, str):
        try:
            # 文字列 "{'X1': ...}" を本物の辞書に変換
            raw_data = ast.literal_eval(raw_data)
            
        except:
            raw_data = json.loads(raw_data.replace("'", '"'))
    
    # 2. 順番通りに値だけを抽出するためのキーのリスト
    feature_keys = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    
    # 3. 値だけを入れるリストを作成し、フラット化する
    feature_values = []

    missing_keys = [] # 不足している変数をメモするリスト
    
    for key in feature_keys:
        val = raw_data.get(key)
        # LLMがリストで送ってきても、最初の値を取り出す

        # 値が存在しない（None）かチェック
        if val is None:
            missing_keys.append(key)
        else:
            if isinstance(val, list) and len(val) > 0:
                feature_values.append(val[0])
            else:
                feature_values.append(val)
                
    # 【追加した安全装置】不足しているキーがあれば、エラーメッセージとしてGeminiに返す
    if len(missing_keys) > 0:
        return f"予測を実行できません。以下の変数が不足しています: {missing_keys}。ユーザーにこれらの値を聞いてください。"
            
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID_NW)
        
        # 4. 2次元配列として送信
        print(f"DEBUG - NW Final Sending Instance: {feature_values}")
        response = endpoint.predict(instances=[feature_values])
        
        # response.predictions[0] で予測値そのものを取り出して返す
        return f"NWモデルの計算結果: {response.predictions[0]}"
    
    except Exception as e:
        return f"NWツール実行中にエラーが発生しました: {str(e)}"

def predict_Gauss_model(input_data=None, **kwargs) -> str:
    """ガウスモデルを使用して数値を算出します。"""
    raw_data = input_data if input_data else kwargs
    # LLMが気まぐれに「文字列」で送ってきた場合の救済措置
    if isinstance(raw_data, str):
        try:
            # 文字列 "{'X1': ...}" を本物の辞書に変換
            raw_data = ast.literal_eval(raw_data)
        except:
            raw_data = json.loads(raw_data.replace("'", '"'))
    
    # 1. 順番通りに値だけを抽出するためのキーのリスト
    feature_keys = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    
    # 2. 値だけを入れるリストを作成
    feature_values = []
    
    missing_keys = [] # 不足している変数をメモするリスト
    
    for key in feature_keys:
        # 辞書から値を取得（なければ0などを入れることも可能ですが、今回はそのまま取得）
        val = raw_data.get(key)

         # 値が存在しない（None）かチェック
        if val is None:
            missing_keys.append(key)
        else:
            # LLMが {'X1': [0.8]} のようにリストで送ってきても、最初の値を取り出す
            if isinstance(val, list) and len(val) > 0:
                feature_values.append(val[0])
            else:
                feature_values.append(val)

    # 【追加した安全装置】不足しているキーがあれば、エラーメッセージとしてGeminiに返す
    if len(missing_keys) > 0:
        return f"予測を実行できません。以下の変数が不足しています: {missing_keys}。ユーザーにこれらの値を聞いてください。"
    
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        endpoint = aiplatform.Endpoint(ENDPOINT_ID_GAUSS)
        
        # 3. リストのリスト（2次元配列）として送信
        # これにより [[0.8, 808.5, 318.2, ...]] という形式で送信されます
        print(f"DEBUG - Final Sending Instance: {feature_values}")
        response = endpoint.predict(instances=[feature_values])
        
        # response.predictions[0] で予測値そのものを取り出して返す
        return f"ガウスモデルの計算結果: {response.predictions[0]}"
    
    except Exception as e:
        return f"ツール実行中にエラーが発生しました: {str(e)}"

def calculate_inverse_features(target_y: float, target_features: list, fixed_values: dict, model_name: str = "nw") -> str:
    """
    目標となる予測結果（Y）から、複数（または1つ）の変数（X）の最適な組み合わせを逆算します。
    
    target_y: 目標とするYの数値 (例: 16.47)
    target_features: 逆算したい変数名のリスト (例: ['X7', 'X8'])
    fixed_values: 固定する変数の値が入った辞書 (例: {'X1': 0.62, 'X2': 808.5, ...})
    model_name: 使用する予測モデル ('linear', 'nw', 'gauss' のいずれか。デフォルトは 'nw')
    """
    try:
        # 1. LLMの気まぐれデータ型対策（防壁）
        if isinstance(fixed_values, str):
            try:
                fixed_values = ast.literal_eval(fixed_values)
            except:
                fixed_values = json.loads(fixed_values.replace("'", '"'))
                
        if isinstance(target_features, str):
            # 万が一文字列で ['X7', 'X8'] や 'X7' と来た場合の対策
            try:
                target_features = ast.literal_eval(target_features)
            except:
                target_features = [target_features]
                
        # 2. モデル名の指定に応じてエンドポイントをスイッチング
        endpoints_map = {
            "linear": ENDPOINT_ID_LINEAR,
            "nw": ENDPOINT_ID_NW,
            "gauss": ENDPOINT_ID_GAUSS
        }
        # 小文字に変換してマッチング。存在しないモデル名ならNWをデフォルトにする
        target_endpoint_id = endpoints_map.get(model_name.lower(), ENDPOINT_ID_NW)
        
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        endpoint = aiplatform.Endpoint(target_endpoint_id)
        
        ordered_features = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
        
        # 3. 動かす変数の数に合わせて、初期値と探索範囲を自動生成
        num_targets = len(target_features)
        x0 = [5.0] * num_targets  # 初期値5.0からスタート
        bounds = [(0.0, 1000.0)] * num_targets  # 0〜1000の範囲で探索
        
        # 4. 目的関数の定義
        def objective_multi(x_vals):
            input_dict = fixed_values.copy()
            # 動かしたい複数の変数を、それぞれ現在の探索値(x_vals)で上書き
            for feat, val in zip(target_features, x_vals):
                input_dict[feat] = val
            
            input_row = [float(input_dict.get(f, 0.0)) for f in ordered_features]
            y_pred = endpoint.predict(instances=[input_row]).predictions[0]
            
            return (y_pred - target_y) ** 2
            
        # 5. 最適化の実行 (ここで数秒〜十数秒かかります)
        res = minimize(objective_multi, x0, bounds=bounds, method='L-BFGS-B')
        
        # 6. 結果のフォーマット
        if res.success:
            result_text = f"【逆算結果 ({model_name.upper()}モデル使用)】\n"
            result_text += f"目標値 Y={target_y} を達成するための最適化結果:\n"
            final_input = []
            
            for f in ordered_features:
                if f in target_features:
                    idx = target_features.index(f)
                    opt_val = res.x[idx]
                    final_input.append(opt_val)
                    result_text += f" - {f}: {opt_val:.4f} に設定を推奨\n"
                else:
                    final_input.append(float(fixed_values.get(f, 0.0)))
                    
            final_pred = endpoint.predict(instances=[final_input]).predictions[0]
            result_text += f"\n(この設定時の最終予測値 Y={final_pred:.4f})"
            
            return result_text
        else:
            return "最適化エラー: 計算が収束しませんでした。別の条件で試してください。"
            
    except Exception as e:
        return f"逆算ツールの実行中にエラーが発生しました: {str(e)}"
