def get_net(identifier, net_params):
    model = None
    print(identifier)
    if identifier == 'resnet101':
        from net.resnet101 import ResNet101 as Net
    elif identifier == 'xceptionv3':
        from net.xception import Xception as Net
    elif identifier == 'inceptionv3':
        from net.inception_v3 import Inception3 as Net
    else:
        raise ValueError("Model [%s] not recognized." % identifier)

    model = Net(in_shape=net_params["in_shape"], num_classes=net_params["num_classes"])
    print("model [%s] was created" % (model.name()))
    return model