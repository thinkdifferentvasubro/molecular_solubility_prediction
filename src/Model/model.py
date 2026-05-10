from chemprop import models, nn

def dmpnn(
    scale=None,
    depth=None,
    dropout=None,
    dropout_fnn=None,
    d_h=None,
    hidden_dim=None,
    n_layers=None,
    mol_feat_size = None
):
    mp = nn.BondMessagePassing(
        depth=depth,
        dropout=dropout,
        d_h=d_h
    )

    ffn_input_dim = mp.output_dim + mol_feat_size

    agg = nn.MeanAggregation()

    output_transform = None
    if scale is not None:
        output_transform = nn.UnscaleTransform.from_standard_scaler(scale)

    ffn = nn.RegressionFFN(
        input_dim=ffn_input_dim,
        output_transform=output_transform,
        dropout=dropout_fnn,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    )

    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    model = models.MPNN(mp, agg, ffn, True, metric_list)
    return model