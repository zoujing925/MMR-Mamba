"""
2023/10/16,
preprocess kspace data with the undersampling mask in the fastMRI project.
"""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


def create_mask_for_mask_type(mask_type_str, center_fractions, accelerations):
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    else:
        raise Exception(f"{mask_type_str} not supported")


## mri related
def mri_fourier_transform_2d(image, mask):
    '''
  image: input tensor [B, H, W, C]
  mask: mask tensor [H, W]
  '''
    spectrum = torch.fft.fftn(image, dim=(1, 2), norm='ortho')
    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))
    # Downsample k-space
    masked_spectrum = spectrum * mask[None, :, :, None]
    return spectrum, masked_spectrum


## mri related
def mri_inver_fourier_transform_2d(spectrum):
    '''
  image: input tensor [B, H, W, C]
  '''
    spectrum = torch.fft.ifftshift(spectrum, dim=(1, 2))
    image = torch.fft.ifftn(spectrum, dim=(1, 2), norm='ortho')

    return image


def add_gaussian_noise(kspace, snr):
    ### 根据SNR确定noise的放大比例
    num_pixels = kspace.shape[0]*kspace.shape[1]*kspace.shape[2]*kspace.shape[3]
    psr = torch.sum(torch.abs(kspace.real)**2)/num_pixels
    pnr = psr/(np.power(10, snr/10))
    noise_r = torch.randn_like(kspace.real)*np.sqrt(pnr)

    psim = torch.sum(torch.abs(kspace.imag)**2)/num_pixels
    pnim = psim/(np.power(10, snr/10))
    noise_im = torch.randn_like(kspace.imag)*np.sqrt(pnim)

    noise = noise_r + 1j*noise_im
    noisy_kspace = kspace + noise

    return noisy_kspace

 
def mri_fft(raw_mri, _SNR):
    mri = torch.tensor(raw_mri)[None, :, :, None].to(torch.float32)
    spectrum = torch.fft.fftn(mri, dim=(1, 2), norm='ortho')
    # K-space spectrum has been shifted to shift the zero-frequency component to the center of the spectrum
    kspace = torch.fft.fftshift(spectrum, dim=(1, 2))

    if _SNR > 0:
        noisy_kspace = add_gaussian_noise(kspace, _SNR)
    else:
        noisy_kspace = kspace

    noisy_mri = mri_inver_fourier_transform_2d(noisy_kspace)
    noisy_mri = torch.sqrt(torch.real(noisy_mri)**2 + torch.imag(noisy_mri)**2)

    return noisy_kspace[0].permute(2, 0, 1), noisy_mri[0].permute(2, 0, 1), \
        kspace[0].permute(2, 0, 1), mri[0].permute(2, 0, 1)



def undersample_mri(raw_mri, _MRIDOWN, _SNR):
    mri = torch.tensor(raw_mri)[None, :, :, None].to(torch.float32)
    if _MRIDOWN == "4X":
        mask_type_str, center_fraction, MRIDOWN = "random", 0.1, 4
    elif _MRIDOWN == "8X":
        mask_type_str, center_fraction, MRIDOWN = "equispaced", 0.04, 8

    ff = create_mask_for_mask_type(mask_type_str, [center_fraction], [MRIDOWN]) ## 0.2 for MRIDOWN=2, 0.1 for MRIDOWN=4, 0.04 for MRIDOWN=8

    shape = [240, 240, 1]
    mask = ff(shape, seed=1337)
    mask = mask[:, :, 0] # [1, 240]
    # print("mask:", mask.shape)
    # print("original MRI:", mri)

    # print("original MRI:", mri.shape)
    ### under-sample the kspace data.
    kspace, masked_kspace = mri_fourier_transform_2d(mri, mask)
    ### add low-field noise to the kspace data.
    if _SNR > 0:
        noisy_kspace = add_gaussian_noise(masked_kspace, _SNR)
    else:
        noisy_kspace = masked_kspace

    ### conver the corrupted kspace data back to noisy MRI image.
    noisy_mri = mri_inver_fourier_transform_2d(noisy_kspace)
    noisy_mri = torch.sqrt(torch.real(noisy_mri)**2 + torch.imag(noisy_mri)**2)

    return noisy_kspace[0].permute(2, 0, 1), noisy_mri[0].permute(2, 0, 1), \
        kspace[0].permute(2, 0, 1), mri[0].permute(2, 0, 1), mask.unsqueeze(-1)



class MaskFunc(object):
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be
                retained. If multiple values are provided, then one of these
                numbers is chosen uniformly each time. 
            accelerations (List[int]): Amount of under-sampling. This should have
                the same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(self, shape, seed=None):
        """
        Create the mask.

        Args:
            shape (iterable[int]): The shape of the mask to be created. The
                shape should have at least 3 dimensions. Samples are drawn
                along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting
                the seed ensures the same mask is generated each time for the
                same shape. The random state is reset afterwards.
                
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data. 
    """

    def __call__(self, shape, seed):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The
                shape should have at least 3 dimensions. Samples are drawn
                along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting
                the seed ensures the same mask is generated each time for the
                same shape. The random state is reset afterwards.

        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask
