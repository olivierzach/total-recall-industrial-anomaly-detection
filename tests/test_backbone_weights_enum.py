from src.patchcore.backbone import _weights_enum_name


def test_weights_enum_name_common_models():
    assert _weights_enum_name("wide_resnet50_2") == "Wide_ResNet50_2_Weights"
    assert _weights_enum_name("vit_b_16") == "ViT_B_16_Weights"
