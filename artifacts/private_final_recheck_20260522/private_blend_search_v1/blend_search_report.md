# Private Blend Search

All candidates are deterministic val/test blends; no Kaggle public feedback is used.

## Baselines

- Current pair `intersect_bold7h_33028` + `widebankG_hailmary_30702`: p10 `0.517802`, mean `0.558186`, regret `0.003269`
- Old pair `intersect_bold7h_33028` + `fusion_samesrc03_32274`: p10 `0.516823`, mean `0.556049`, regret `0.005406`

## Top Generated Candidates

| Candidate | Pair P10 | Pair Mean | Regret | Delta P10 | J Anchor | J Public | Cites | Recipe |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `vote_winners_private_t24` | `0.539120` | `0.572799` | `-0.011345` | `0.021318` | `0.878515` | `0.925759` | `889` | vote pool=winners weights=private threshold=0.24 |
| `base_widebankG_hailmary_30702_winners_uniform_k18_a42` | `0.539120` | `0.572799` | `-0.011345` | `0.021318` | `0.881490` | `0.926719` | `886` | base=widebankG_hailmary_30702 pool=winners weights=uniform keep=0.18 add=0.42 |
| `base_widebankG_hailmary_30702_samesrc_diverse_k10_a70` | `0.538094` | `0.572298` | `-0.010843` | `0.020292` | `0.888510` | `0.925339` | `879` | base=widebankG_hailmary_30702 pool=samesrc weights=diverse keep=0.10 add=0.70 |
| `vote_nonclone_private_t60` | `0.537692` | `0.575521` | `-0.014066` | `0.019890` | `0.934211` | `0.887372` | `836` | vote pool=nonclone weights=private threshold=0.60 |
| `base_widebankG_hailmary_30702_winners_private_k24_a70` | `0.537692` | `0.575521` | `-0.014066` | `0.019890` | `0.935329` | `0.888383` | `835` | base=widebankG_hailmary_30702 pool=winners weights=private keep=0.24 add=0.70 |
| `base_widebankG_hailmary_30702_winners_uniform_k18_a50` | `0.536770` | `0.574824` | `-0.013369` | `0.018968` | `0.890536` | `0.929625` | `877` | base=widebankG_hailmary_30702 pool=winners weights=uniform keep=0.18 add=0.50 |
| `base_widebankG_hailmary_30702_top16_uniform_k24_a50` | `0.536770` | `0.574824` | `-0.013369` | `0.018968` | `0.891553` | `0.928490` | `876` | base=widebankG_hailmary_30702 pool=top16 weights=uniform keep=0.24 add=0.50 |
| `base_widebankG_hailmary_30702_winners_p10_k18_a50` | `0.536770` | `0.574824` | `-0.013369` | `0.018968` | `0.892571` | `0.929545` | `875` | base=widebankG_hailmary_30702 pool=winners weights=p10 keep=0.18 add=0.50 |
| `base_intersect_bold7h_33028_nonclone_diverse_k00_a70` | `0.535946` | `0.573967` | `-0.012512` | `0.018144` | `0.936451` | `0.889396` | `834` | base=intersect_bold7h_33028 pool=nonclone weights=diverse keep=0.00 add=0.70 |
| `base_fusion_samesrc03_32274_top6_diverse_k24_a70` | `0.535946` | `0.573967` | `-0.012512` | `0.018144` | `0.937575` | `0.890411` | `833` | base=fusion_samesrc03_32274 pool=top6 weights=diverse keep=0.24 add=0.70 |
| `vote_top16_diverse_t50` | `0.535946` | `0.573967` | `-0.012512` | `0.018144` | `0.938702` | `0.891429` | `832` | vote pool=top16 weights=diverse threshold=0.50 |
| `base_intersect_bold7h_33028_nonclone_p10_k00_a70` | `0.535946` | `0.573967` | `-0.012512` | `0.018144` | `0.940964` | `0.893471` | `830` | base=intersect_bold7h_33028 pool=nonclone weights=p10 keep=0.00 add=0.70 |
| `base_fusion_samesrc03_32274_top10_diverse_k30_a70` | `0.535946` | `0.573967` | `-0.012512` | `0.018144` | `0.944377` | `0.896552` | `827` | base=fusion_samesrc03_32274 pool=top10 weights=diverse keep=0.30 add=0.70 |
| `base_widebankG_hailmary_30702_samesrc_diverse_k10_a60` | `0.535763` | `0.570470` | `-0.009015` | `0.017961` | `0.887500` | `0.924294` | `880` | base=widebankG_hailmary_30702 pool=samesrc weights=diverse keep=0.10 add=0.60 |
| `base_widebankG_hailmary_30702_samesrc_diverse_k10_a42` | `0.535763` | `0.569650` | `-0.008195` | `0.017961` | `0.884485` | `0.927684` | `883` | base=widebankG_hailmary_30702 pool=samesrc weights=diverse keep=0.10 add=0.42 |
| `vote_top16_uniform_t30` | `0.535582` | `0.570030` | `-0.008575` | `0.017780` | `0.883484` | `0.930995` | `884` | vote pool=top16 weights=uniform threshold=0.30 |
| `vote_top16_private_t30` | `0.535582` | `0.570030` | `-0.008575` | `0.017780` | `0.884485` | `0.929864` | `883` | vote pool=top16 weights=private threshold=0.30 |
| `base_widebankG_hailmary_30702_winners_uniform_k24_a42` | `0.535582` | `0.570030` | `-0.008575` | `0.017780` | `0.886493` | `0.931973` | `881` | base=widebankG_hailmary_30702 pool=winners weights=uniform keep=0.24 add=0.42 |
| `base_widebankG_hailmary_30702_winners_uniform_k24_a50` | `0.535285` | `0.571959` | `-0.010504` | `0.017483` | `0.895642` | `0.934932` | `872` | base=widebankG_hailmary_30702 pool=winners weights=uniform keep=0.24 add=0.50 |
| `base_widebankG_hailmary_30702_top16_uniform_k30_a50` | `0.535285` | `0.571959` | `-0.010504` | `0.017483` | `0.896670` | `0.933790` | `871` | base=widebankG_hailmary_30702 pool=top16 weights=uniform keep=0.30 add=0.50 |
