"""WDCNN: as formas da Tabela 2 de zhang2017wdcnn (conferida no PMC5336047)."""

import torch
from snq_learn import WDCNN

TABLE2 = [  # (camada, (largura, canais)) da Tabela 2
    ("conv1", (128, 16)),
    ("pool1", (64, 16)),
    ("conv2", (64, 32)),
    ("pool2", (32, 32)),
    ("conv3", (32, 64)),
    ("pool3", (16, 64)),
    ("conv4", (16, 64)),
    ("pool4", (8, 64)),
    ("conv5", (6, 64)),
    ("pool5", (3, 64)),
    ("fc", (100, 1)),
]


def test_formas_batem_com_a_tabela_2():
    assert WDCNN(n_classes=10).trace_shapes() == TABLE2


def test_saida_tem_uma_coluna_por_classe():
    model = WDCNN(n_classes=10).eval()
    with torch.no_grad():
        assert model(torch.randn(4, 1, 2048)).shape == (4, 10)
        assert model(torch.randn(4, 2048)).shape == (4, 10)  # aceita (B, L)
    assert WDCNN(n_classes=2)(torch.randn(3, 2048)).shape == (3, 2)


def test_primeira_camada_e_larga_e_cada_convolucao_tem_bn():
    model = WDCNN(n_classes=10)
    convs = [m for m in model.features if isinstance(m, torch.nn.Conv1d)]
    assert (convs[0].kernel_size, convs[0].stride, convs[0].out_channels) == ((64,), (16,), 16)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(bns) == 6  # cinco convolucoes e a totalmente conectada
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
