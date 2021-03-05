from BESD.neural_models.models_convolutional import build_conv_with_fusion

def build_model(*args, **kwargs):
    model = build_conv_with_fusion(*args, **kwargs)
    return model
