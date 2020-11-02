from monai.transforms import Compose

def split_train_val(data_list):
    split_point = int(len(data_list)*0.7)
    data_train = data_list[:split_point]
    data_val = data_list[split_point:]

    return data_val, data_val


class TrainCompose(Compose):
    def __call__(self, input_):
        # load nifti
        vol = self.transforms[0](input_)
        vol = vol[..., 0]
        # add channel
        vol = self.transforms[1](vol)
        # rotation
        vol = self.transforms[2](vol)
        vol = self.transforms[3](vol)
        # flip
        vol = self.transforms[4](vol)
        #vol = self.transforms[5](vol)
        #vol = self.transforms[6](vol)
        return vol

class ValCompose(Compose):
    def __call__(self, input_):
        # load nifti
        vol = self.transforms[0](input_)
        vol = vol[..., 0]  # selecting magnitude
        # add channel
        vol = self.transforms[1](vol)
        # rotation
        vol = self.transforms[2](vol)
        return vol

