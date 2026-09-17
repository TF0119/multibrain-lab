# Backlog

| Item | Why | When |
|------|-----|------|
| hud-visual-check-in-browser | 活動 HUD（バー、スパークライン、点群）はビルドと復号試験のみで、ブラウザでの描画を目視していない。`stream_body.py --mode wave --activity --brains 3 --speed 1` と `npm run dev` で確認する | いつでも可 |
| g1-training-run | `train_mlp.py --nworld 256 --total-steps 200000000` で G1 を回し、報酬係数と §3.2 の上限トルクを固定する。256 環境で約 1.5 時間 | 抜け道試験と PPO の補助値の確認の後 |
| m0-contact-remeasure | M0 の接触数と速度は右側体節が左と重なる旧身体で測った値。修正後の身体で `check_body.py` を再実行し §3.4 と §3.6 を更新する | いつでも可 |
| reward-action-arg | `Reward.step` の action に env がクランプ済み指令を渡している。tanh 方策では同一だが、神経中核の読出しを繋ぐときは方策の生の出力を渡す設計に改める | M2 で神経中核を環境に繋ぐとき |
| efc-limit-force-logging | 関節限界の受動拘束力（`efc_force`）の記録が未実装（§5.3 は記録だけすると定める） | M3 の監視記録を作るとき |
