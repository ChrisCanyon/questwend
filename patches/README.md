# third_party/ggml に載っているパッチ

`third_party/ggml` は upstream の [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
の `ggml/` を vendor したもの（コミットは `third_party/ggml/GGML_VERSION`）で、
**現在 qwencpp 固有のパッチは無い**（upstream そのまま）。

2026-10-14 の再 vendor（`1b70350c9`）で、以前当てていた 2 本は不要になったため削除した:

| # | 旧パッチ | 不要になった理由 |
|---|---|---|
| 01 | `01-qwencpp-metal-mmid-blocked.patch` | Metal `mm_id_map0` が expert 数 `ne02` > threadgroup 上限で assert する問題。upstream の `ggml_metal_op_mul_mat_id_use_mm` が `ne02 <= 1024` を条件に含むようになり、それを超える expert キャッシュのスロットプールは mat-vec 経路へ自動で回る。A/B スイッチ `QWEN_METAL_MMID_MV` は無くなった。**未検証**: pipeline のスレッド上限が 1024 未満で `ne02` がその間に入る場合は upstream の assert が残るので、Mac で一度確認すること |
| 02 | `02-qwencpp-cuda-quiet-graph-logs.patch` | CUDA graph の DEBUG ログ。upstream が `ggml_cuda_graph_debug_enabled()` の下に入れたので既定では出ない |

パッチを再び足すときは `patches/0N-*.patch` に置き、下の手順 3 で当てる。

> サブ 1bit 量子化（`IQ1_XS` / `IQ1_XXS` / `IQ1_XXXS`）は **`iq1-narrow` ブランチ**に分けてあります。
> 本家 llama.cpp に取り込まれたら、そのブランチを捨てて通常の再 vendor で拾えば済みます。

## 再 vendor の手順

```bash
# 1. upstream を取得（例: ローカルの llama.cpp クローンから）
cd /path/to/llama.cpp && git fetch origin master

# 2. ggml/ を差し替え（古い vendor の残骸を消すため必ず rm -rf してから展開する）
cd /path/to/qwencpp
rm -rf third_party/ggml
(cd /path/to/llama.cpp && git archive origin/master ggml) | tar -x -C third_party
echo "Vendored from https://github.com/ggml-org/llama.cpp ggml/ at commit $(git -C /path/to/llama.cpp rev-parse origin/master)" \
  > third_party/ggml/GGML_VERSION

# 3. パッチを番号順に適用（リポジトリルートから、--directory に注意）
for p in patches/0*.patch; do git apply --directory=third_party -p1 "$p" || echo "CONFLICT: $p"; done
```

適用後は `patches/` を再生成して「upstream + パッチ = 現ツリー」を保つこと
（upstream の変更で当たらなくなった場合は手で当て直してから再生成）。

### 落とし穴

- **`git apply` はリポジトリ内のサブディレクトリから実行すると、パッチのパスを
  リポジトリルート基準で解釈し、カレント外のパスを"成功"扱いで黙って読み飛ばす**。
  `cd third_party && git apply -p1 …` は何もせず exit 0 になる。必ず
  **リポジトリルートから `--directory=third_party`** を使うこと（上記手順のとおり）。
- **`rm -rf third_party/ggml` を省いて上書き展開すると、upstream で削除された
  ファイルが残る**（`fattn-wmma-f16.cu` など）。ビルドは通ってしまうことがあるので厄介。
- ggml の DLL は `build/bin/<config>/` に出るが exe は `build/<config>/` に出る。
  exe の隣に古い DLL が残っていると**それが優先されて古い ggml で動く**（実際に踏んだ:
  ソースと合わない行番号の assert が出る）。ルート `CMakeLists.txt` の POST_BUILD
  コピーで同期しているので、手でコピーした DLL を置かないこと。
- upstream の API 変更は qwencpp 側に波及する。今回は `ggml_gated_delta_net` の
  state 入力が平坦な `[S*S*H_v, K, n_seqs]` から初期状態のみの 4D
  `[S, S, H_v, n_seqs]` になり、スナップショット数 `K` が明示引数になった。
