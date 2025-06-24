# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/4/24 16:35
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : stereo_chip_name.py
🌟 Description  : 
🌟 Key Words  :
"""


def get_chip_prefix_info(
        prefix: str,
        chip_name: str = None
):
    """

    Args:
        prefix:
        chip_name:

    Returns:

    """
    raise NotImplementedError("Need (stereo_chip_name.pyd) file.")


if __name__ == '__main__':
    from stereo_chip_name import get_chip_prefix_info

    scn = get_chip_prefix_info('A')

    print(scn)



