"""Map of the Portuguese labels recorded in result files to their English names (2026-11-13).

The labels were renamed after the runs so that the whole public release is in English. The
renaming is a pure substitution: `tools/relabel_results.py` applies it, records what it changed in
every file, and can apply the inverse to recover the original bytes, whose hashes the manifests and
analysis files recorded.

- TOKENS: whole labels, replaced only where they are delimited by characters other than letters,
  digits and underscores (so `dec_aprendida` never touches `dec_aprendida_sel`).
- SENTENCES: free-text values (skip reasons, manifest descriptions), as regular expressions.
- PATHS: directories and files whose names carry a label.
- SEED_TAGS: literals that feed seed derivation. They are never renamed, and the code derives
  seeds from the legacy label of every renamed arm, so reruns draw the same random numbers.
"""

from __future__ import annotations

TOKENS: dict[str, str] = {
    # streams and cleaning
    "bruto": "raw",
    "mediana3": "median3",
    "sem_picos": "despiked",
    "despike_decimado": "despike_after_decimation",
    "mediana3_decimado": "median3_after_decimation",
    # selectors and arms (E20, E24, E26)
    "kurtograma": "kurtogram",
    "kurtograma_sem_picos": "kurtogram_despiked",
    "kurtograma_sem_picos_decimado": "kurtogram_despiked_after_decimation",
    "kurtograma_teto1k": "kurtogram_ceiling1k",
    "kurtograma_teto2k": "kurtogram_ceiling2k",
    "kurtograma_teto4k": "kurtogram_ceiling4k",
    "kurtograma_teto8k": "kurtogram_ceiling8k",
    "kurtograma_teto": "kurtogram_ceiling",
    "kurtograma_notch": "kurtogram_notch",
    "kurtograma_notch_controle": "kurtogram_notch_control",
    "kurtograma_regra": "kurtogram_rule",
    "mediana_kurtograma": "median_kurtogram",
    "guiado": "guided",
    "mediana_guiado": "median_guided",
    "cego": "blind",
    "infograma_ses": "ses_infogram",
    "infograma_ses_despike": "ses_infogram_despike",
    "infograma_ses_sem_picos": "ses_infogram_despiked",
    "infograma_ses_regra": "ses_infogram_rule",
    "infograma_ses_teto1k": "ses_infogram_ceiling1k",
    "infograma_ses_teto2k": "ses_infogram_ceiling2k",
    "infograma_ses_teto4k": "ses_infogram_ceiling4k",
    "infograma_ses_teto8k": "ses_infogram_ceiling8k",
    "banda_larga": "wideband",
    "banda_larga_regra": "wideband_rule",
    "sorteada": "random_band",
    "sorteada_regra": "random_band_rule",
    "pura": "pure",
    "zona": "zone",
    "zona1": "zone1",
    "zona2": "zone2",
    "zona3": "zone3",
    "zona4": "zone4",
    "zona5": "zone5",
    "zona6": "zone6",
    "zona7": "zone7",
    "zona_frota": "fleet_zone",
    "zona_kurtograma": "kurtogram_zone",
    "_regra": "_rule",
    "_teto": "_ceiling",
    "teto": "ceiling",
    "teto1k_vs_32x": "ceiling1k_vs_32x",
    "teto2k_vs_16x": "ceiling2k_vs_16x",
    "teto4k_vs_8x": "ceiling4k_vs_8x",
    "teto8k_vs_4x": "ceiling8k_vs_4x",
    "teto1k_vs_sem_teto": "ceiling1k_vs_no_ceiling",
    "teto2k_vs_sem_teto": "ceiling2k_vs_no_ceiling",
    "teto4k_vs_sem_teto": "ceiling4k_vs_no_ceiling",
    "teto8k_vs_sem_teto": "ceiling8k_vs_no_ceiling",
    "k_vs_sem_teto": "k_vs_no_ceiling",
    # decisions and races
    "nenhuma": "no_fault",
    "saudavel": "healthy_bearing",
    # rulers and calibration
    "aninhada": "nested",
    "propria_calibracao": "own_calibration",
    "indefinida": "undefined",
    "regua": "ruler",
    "calibracao": "calibration",
    # estimation modes (E07)
    "sem_prior": "no_prior",
    "com_prior": "with_prior",
    "prior_calibrado": "calibrated_prior",
    "prior_nominal": "nominal_prior",
    # uniform and classical arms (E15, E22, E23)
    "uniforme": "uniform",
    "uniforme_teto": "uniform_ceiling",
    "uniforme_grid": "uniform_grid",
    "uniforme_4": "uniform_4",
    "uniforme_orcamento": "uniform_budget",
    "uniforme_treino": "uniform_trained",
    "uniforme_treino_por_grupo": "uniform_trained_per_race",
    "oraculo": "oracle",
    "oraculo_uniforme": "uniform_oracle",
    "oraculo_uniforme_por_grupo": "uniform_oracle_per_race",
    "uniforme_varredura_cv": "uniform_sweep_cv",
    "uniforme_varredura_registro": "uniform_sweep_record",
    "uniforme_varredura_registro_vazamento": "uniform_sweep_record_leak",
    "varredura_registro": "sweep_record",
    "noise_amplification_uniforme": "noise_amplification_uniform",
    "taxa_tat": "tat_rate",
    "taxa_cheia": "full_rate",
    "aleatorio_cs": "random_cs",
    "passa_faixa_lu": "bandpass_lu",
    "multitaxa": "multirate",
    "aleatorio_cs_vs_uniforme": "random_cs_vs_uniform",
    "passa_faixa_lu_vs_uniforme": "bandpass_lu_vs_uniform",
    "taxa_tat_vs_uniforme": "tat_rate_vs_uniform",
    "uniforme_teto_vs_uniforme": "uniform_ceiling_vs_uniform",
    # learned-mask arms (E15)
    "dec_uniforme": "dec_uniform",
    "dec_uniforme_treino": "dec_uniform_trained",
    "dec_aleatoria": "dec_random",
    "dec_aprendida": "dec_learned",
    "dec_aprendida_sel": "dec_learned_sel",
    "dec_aprendida_sem_ajuste": "dec_learned_norefit",
    "dec_livre": "dec_free",
    "dec_livre_projetada": "dec_free_projected",
    "dec_livre_projetada_sem_ajuste": "dec_free_projected_norefit",
    "dec_loupe_livre": "dec_loupe_free",
    "dec_loupe_projetada": "dec_loupe_projected",
    "dec_loupe_projetada_sem_ajuste": "dec_loupe_projected_norefit",
    "dec_adaptativa": "dec_adaptive",
    "dec_adaptativa_blocos": "dec_adaptive_blocks",
    "dec_aprendida_A": "dec_learned_A",  # stage-A labels of the training history (E15 runs/)
    "dec_livre_A": "dec_free_A",
    "dec_adaptativa_A": "dec_adaptive_A",
    "livre": "free",  # config block of the free mask
    "aprendida": "learned",
    "livre_projetada": "free_projected",
    "com_raia": "with_line",
    "sem_raia": "without_line",
    "todos": "all",
    "perfeito": "perfect",
    "otimista": "optimistic",
    "otim": "optimistic",  # internal key of the power gate for the optimistic rule
    "pre_registrada": "preregistered",
    "testavel": "testable",
    "nao_testavel": "not_testable",
    "ilegivel": "unreadable",
    # fault positions (E23)
    "centrada": "centred",
    "oposta": "opposite",
    "ortogonal": "orthogonal",
    # verdicts and analysis keys
    "apoiada": "supported",
    "refutada": "refuted",
    "inconclusiva": "inconclusive",
    "inconclusivo": "inconclusive",
    "recupera": "recovers",
    "D3_registro_anticonservador": "D3_record_anticonservative",
    "posthoc_tat_minus_registro": "posthoc_tat_minus_record",
    "H1_cs_minus_teto": "H1_cs_minus_ceiling",
    "H1_linha_impulsividade": "H1_line_impulsiveness",
    "H1a_sensibilidade": "H1a_sensitivity",
    "H1b_especificidade": "H1b_specificity",
    "H2_tat_vs_oraculo": "H2_tat_vs_oracle",
    "verdict_H_picos": "verdict_H_spikes",
    "orcamento": "budget",
    "registros": "records",
    "pistas": "races",
    "retomado": "resumed",
    "guarda": "guard",
    # co-prime pilot (E25)
    "coprimo_8_9": "coprime_8_9",
    # E22 analysis printout
    "varredura_cv": "sweep_cv",
    # header of the E15 convergence table (results-lr30)
    "variante": "variant",
    "divisao": "split",
    "convergiu": "converged",
    "margem": "margin",
    "melhor_epoca": "best_epoch",
    "epocas": "epochs",
    "posicoes": "positions",
}

# fmt: off
# (regular expression, replacement), applied to whole free-text values before TOKENS
SENTENCES: list[tuple[str, str]] = [
    (r"^(\S+\.mat): rotação com ([\d.]+) s, esperado 4 s$",
     r"\1: rotation signal of \2 s, expected 4 s"),
    (r"^(\S+\.mat): vibração com (\d+) amostras, menos de 4 s a 64 kHz$",
     r"\1: vibration with \2 samples, less than 4 s at 64 kHz"),
    (r"^(\S+\.mat): vibração com (\d+) amostras, menos de (\d+)$",
     r"\1: vibration with \2 samples, fewer than \3"),
    (r"^(\S+\.mat): ilegível pelo scipy \((.*)\)$", r"\1: unreadable by scipy (\2)"),
    (r"^sha256 sobre a concatenacao, em ordem de caminho POSIX, de '<caminho relativo ao "
     r"repositorio>\\n<sha256 do conteudo do arquivo>\\n' para cada (\S+)$",
     r"sha256 over the concatenation, in POSIX path order, of '<path relative to the "
     r"repository>\\n<sha256 of the file content>\\n' for each \1"),
    (r"^MAFAULDA \(UFRJ\), licenca nao declarada; fs inferida 51,2 kHz \(ver manifest do "
     r"dataset\)$",
     "MAFAULDA (UFRJ), license not declared; inferred fs 51.2 kHz (see the dataset manifest)"),
    (r"^dicionario (\d+x\d+) acima do teto$", r"dictionary \1 above the ceiling"),
    (r"^acuracia balanceada de ordem \(media sobre as pistas do acerto \|est - regua\| < 0,06\)$",
     "balanced order accuracy (mean over races of the hit |est - ruler| < 0.06)"),
    (r"^raia do proprio trecho em taxa cheia \(snq_recon\.refine_order, centro do plato\); grade "
     r"deslizante para treino e validacao, segmento para escolha e pontuacao$",
     "full-rate line of the segment itself (snq_recon.refine_order, centre of the plateau); "
     "sliding grid for training and validation, segment for selection and scoring"),
    (r"^mediana das calibracoes do segmento 0 dos registros fora do teste do grupo$",
     "median of the segment-0 calibrations of the records outside the test set of the group"),
    (r"^determinado empiricamente; nao documentado pelos autores$",
     "determined empirically; not documented by the authors"),
    (r"^dobra refutada como causa do piso$", "folding refuted as the cause of the floor choice"),
    (r"^teto nao alivia$", "the ceiling does not relieve the capture"),
    (r"^nao refutada$", "not refuted"),
    (r"^nao recupera$", "does not recover"),
    (r"^nao avaliavel \(sem P\*\)$", "not evaluable (no P*)"),
    (r"^nao avaliavel \(sem acerto ou sem erro\)$", "not evaluable (no hit or no miss)"),
    (r"^nao avaliavel$", "not evaluable"),
    (r"^sem fator escolhido$", "no factor chosen"),
    (r"^teto abaixo do piso do kurtograma$", "ceiling below the kurtogram floor"),
    (r"^controle invalido$", "invalid control"),
    (r"^nao instalado$", "not installed"),
    (r"^ausente$", "missing"),
    (r"^sem injeção$", "no injection"),
    (r"^sem injeção\|", "no injection|"),
    (r"\(Paderborn, aquisicao\)$", "(Paderborn, acquisition)"),
    (r"\(Paderborn, regra fixa\)$", "(Paderborn, fixed rule)"),
]

PATHS: dict[str, str] = {
    "experiments/E07-order-domain-ottawa/config/smoke-regua.yaml":
        "experiments/E07-order-domain-ottawa/config/smoke-ruler.yaml",
    "experiments/E07-order-domain-ottawa/results-smoke-regua":
        "experiments/E07-order-domain-ottawa/results-smoke-ruler",
    "experiments/E20-fold-zone-rejection/config/smoke-regua.yaml":
        "experiments/E20-fold-zone-rejection/config/smoke-ruler.yaml",
    "experiments/E20-fold-zone-rejection/results-smoke-regua":
        "experiments/E20-fold-zone-rejection/results-smoke-ruler",
    "experiments/E24-paderborn-spike-capture/config/teto_notch.yaml":
        "experiments/E24-paderborn-spike-capture/config/ceiling_notch.yaml",
    "experiments/E24-paderborn-spike-capture/results-teto-notch":
        "experiments/E24-paderborn-spike-capture/results-ceiling-notch",
    "experiments/E24-paderborn-spike-capture/exploratory/teto_notch_analysis.py":
        "experiments/E24-paderborn-spike-capture/exploratory/ceiling_notch_analysis.py",
    "experiments/E26-fault-order-guided-selector/results-paderborn-regra":
        "experiments/E26-fault-order-guided-selector/results-paderborn-rule",
    "experiments/E26-fault-order-guided-selector/results-paderborn-aquisicao":
        "experiments/E26-fault-order-guided-selector/results-paderborn-acquisition",
    "experiments/E15-learned-sampling-mask/results-lr10/SELADO.md":
        "experiments/E15-learned-sampling-mask/results-lr10/SEALED.md",
    "experiments/E15-learned-sampling-mask/results-lr30/SELADO.md":
        "experiments/E15-learned-sampling-mask/results-lr30/SEALED.md",
    "experiments/E15-learned-sampling-mask/results-lr30/convergencia.txt":
        "experiments/E15-learned-sampling-mask/results-lr30/convergence.txt",
}
# the run cache of the E15 smoke run: runs/<dataset>/<variant>/<budget>/<split>/<seed>/<method>.json
_RUNS = "experiments/E15-learned-sampling-mask/results-smoke/runs/ottawa"
for _var in ("bruto", "sem_picos"):
    for _b in ("b16", "b32"):
        for _m in ("dec_adaptativa", "dec_aleatoria", "dec_aprendida", "dec_livre",
                   "dec_uniforme", "dec_uniforme_treino", "dec_loupe", "dec_multicoset",
                   "dec_multicoset_vb", "wdcnn"):
            PATHS[f"{_RUNS}/{_var}/{_b}/split0/seed0/{_m}.json"] = (
                f"{_RUNS}/{TOKENS[_var]}/{_b}/split0/seed0/{TOKENS.get(_m, _m)}.json"
            )
# fmt: on
# name fragments of PATHS, replaced in text wherever a path is cited
PATH_TOKENS: dict[str, str] = {
    "smoke-regua": "smoke-ruler",
    "teto_notch": "ceiling_notch",
    "teto-notch": "ceiling-notch",
    "results-paderborn-regra": "results-paderborn-rule",
    "results-paderborn-aquisicao": "results-paderborn-acquisition",
    "SELADO.md": "SEALED.md",
    "convergencia.txt": "convergence.txt",
}

# never renamed: they feed seed derivation (see the module docstring)
SEED_TAGS = ("aleatoria", "dados")
