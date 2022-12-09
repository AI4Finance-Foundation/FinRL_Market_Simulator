import os
import numpy as np
import pandas as pd

SharesShangZheng50 = [
    '600000.XSHG', '600015.XSHG', '600016.XSHG', '600028.XSHG', '600030.XSHG', '600036.XSHG', '600050.XSHG',
    '600104.XSHG', '600111.XSHG', '600519.XSHG', '600837.XSHG', '600048.XSHG', '601006.XSHG', '601398.XSHG',
    '601628.XSHG', '601166.XSHG', '601318.XSHG', '601328.XSHG', '601169.XSHG', '601088.XSHG', '601857.XSHG',
    '601601.XSHG', '601668.XSHG', '601288.XSHG', '601818.XSHG', '600585.XSHG', '601766.XSHG', '601989.XSHG',
    '600010.XSHG', '601299.XSHG', '600256.XSHG', '600887.XSHG', '601901.XSHG', '601688.XSHG', '600518.XSHG',
    '600999.XSHG', '600332.XSHG', '600406.XSHG', '600637.XSHG', '600018.XSHG', '600089.XSHG', '600196.XSHG',
    '600703.XSHG', '600832.XSHG', '601118.XSHG', '600109.XSHG', '600150.XSHG', '600372.XSHG', '600690.XSHG',
    '601998.XSHG']
CommonShares497 = [
    '001267_XSHE', '000951_XSHE', '001236_XSHE', '000910_XSHE', '000609_XSHE', '000918_XSHE', '000915_XSHE',
    '000928_XSHE', '000616_XSHE', '000969_XSHE', '000793_XSHE', '000996_XSHE', '000838_XSHE', '001213_XSHE',
    '000767_XSHE', '000961_XSHE', '001209_XSHE', '001229_XSHE', '000408_XSHE', '001230_XSHE', '000034_XSHE',
    '000615_XSHE', '000890_XSHE', '000617_XSHE', '000948_XSHE', '000009_XSHE', '000957_XSHE', '000980_XSHE',
    '001227_XSHE', '000539_XSHE', '001210_XSHE', '000976_XSHE', '000955_XSHE', '000779_XSHE', '000831_XSHE',
    '000090_XSHE', '000593_XSHE', '000901_XSHE', '000017_XSHE', '000007_XSHE', '000987_XSHE', '000688_XSHE',
    '000836_XSHE', '000546_XSHE', '000756_XSHE', '000938_XSHE', '000893_XSHE', '000762_XSHE', '000829_XSHE',
    '000837_XSHE', '000862_XSHE', '000561_XSHE', '000012_XSHE', '000690_XSHE', '000881_XSHE', '000883_XSHE',
    '000572_XSHE', '000006_XSHE', '000560_XSHE', '000917_XSHE', '000058_XSHE', '000710_XSHE', '000156_XSHE',
    '000529_XSHE', '000685_XSHE', '000606_XSHE', '000921_XSHE', '001208_XSHE', '000859_XSHE', '000750_XSHE',
    '000977_XSHE', '000909_XSHE', '000983_XSHE', '000589_XSHE', '000625_XSHE', '000703_XSHE', '000663_XSHE',
    '000554_XSHE', '000036_XSHE', '000409_XSHE', '000835_XSHE', '000985_XSHE', '000919_XSHE', '000818_XSHE',
    '000570_XSHE', '000023_XSHE', '000932_XSHE', '000757_XSHE', '000707_XSHE', '000713_XSHE', '000988_XSHE',
    '000166_XSHE', '001317_XSHE', '000528_XSHE', '000963_XSHE', '000025_XSHE', '000059_XSHE', '000825_XSHE',
    '000623_XSHE', '000778_XSHE', '000069_XSHE', '000014_XSHE', '000048_XSHE', '000958_XSHE', '000712_XSHE',
    '000157_XSHE', '000016_XSHE', '000965_XSHE', '000551_XSHE', '000019_XSHE', '000638_XSHE', '000420_XSHE',
    '000049_XSHE', '000608_XSHE', '000973_XSHE', '000729_XSHE', '000029_XSHE', '000995_XSHE', '000966_XSHE',
    '000159_XSHE', '000989_XSHE', '000708_XSHE', '000927_XSHE', '000815_XSHE', '000429_XSHE', '000600_XSHE',
    '000697_XSHE', '000723_XSHE', '000911_XSHE', '000785_XSHE', '000777_XSHE', '000498_XSHE', '000700_XSHE',
    '000912_XSHE', '000558_XSHE', '000035_XSHE', '000430_XSHE', '000953_XSHE', '001318_XSHE', '000651_XSHE',
    '000510_XSHE', '000563_XSHE', '000887_XSHE', '001205_XSHE', '000567_XSHE', '000026_XSHE', '000526_XSHE',
    '000739_XSHE', '000888_XSHE', '001308_XSHE', '000096_XSHE', '000151_XSHE', '001222_XSHE', '000552_XSHE',
    '000761_XSHE', '000936_XSHE', '000822_XSHE', '000792_XSHE', '000962_XSHE', '000501_XSHE', '000506_XSHE',
    '000839_XSHE', '000791_XSHE', '000933_XSHE', '000400_XSHE', '000982_XSHE', '001266_XSHE', '000858_XSHE',
    '000536_XSHE', '000695_XSHE', '000565_XSHE', '000402_XSHE', '000959_XSHE', '000826_XSHE', '000692_XSHE',
    '000820_XSHE', '000869_XSHE', '000598_XSHE', '000882_XSHE', '000889_XSHE', '000801_XSHE', '000760_XSHE',
    '000920_XSHE', '000100_XSHE', '000425_XSHE', '000828_XSHE', '000517_XSHE', '000070_XSHE', '000715_XSHE',
    '000056_XSHE', '001258_XSHE', '000705_XSHE', '000665_XSHE', '000821_XSHE', '000610_XSHE', '000413_XSHE',
    '000972_XSHE', '000877_XSHE', '000892_XSHE', '000619_XSHE', '001219_XSHE', '000788_XSHE', '000037_XSHE',
    '000421_XSHE', '001234_XSHE', '000753_XSHE', '000553_XSHE', '000810_XSHE', '000923_XSHE', '000591_XSHE',
    '000031_XSHE', '000635_XSHE', '000595_XSHE', '000543_XSHE', '000514_XSHE', '001289_XSHE', '000038_XSHE',
    '000620_XSHE', '000950_XSHE', '001203_XSHE', '000503_XSHE', '000586_XSHE', '000752_XSHE', '000547_XSHE',
    '000411_XSHE', '000733_XSHE', '000878_XSHE', '000564_XSHE', '000672_XSHE', '000720_XSHE', '001313_XSHE',
    '000908_XSHE', '000861_XSHE', '000860_XSHE', '000990_XSHE', '000800_XSHE', '000802_XSHE', '000848_XSHE',
    '000599_XSHE', '000021_XSHE', '000062_XSHE', '000782_XSHE', '001231_XSHE', '000613_XSHE', '000576_XSHE',
    '000718_XSHE', '000537_XSHE', '000153_XSHE', '000780_XSHE', '000895_XSHE', '000799_XSHE', '000679_XSHE',
    '000050_XSHE', '000975_XSHE', '000806_XSHE', '000607_XSHE', '000011_XSHE', '000922_XSHE', '000732_XSHE',
    '000899_XSHE', '000978_XSHE', '000055_XSHE', '000509_XSHE', '001288_XSHE', '000677_XSHE', '000416_XSHE',
    '000655_XSHE', '000585_XSHE', '000885_XSHE', '000789_XSHE', '000981_XSHE', '000531_XSHE', '001228_XSHE',
    '000850_XSHE', '000520_XSHE', '000669_XSHE', '000407_XSHE', '000030_XSHE', '000603_XSHE', '000502_XSHE',
    '000971_XSHE', '000597_XSHE', '000099_XSHE', '000002_XSHE', '000301_XSHE', '000758_XSHE', '000926_XSHE',
    '001215_XSHE', '000629_XSHE', '000929_XSHE', '000630_XSHE', '000656_XSHE', '000573_XSHE', '000590_XSHE',
    '000766_XSHE', '000046_XSHE', '000999_XSHE', '000681_XSHE', '000812_XSHE', '000659_XSHE', '000524_XSHE',
    '000726_XSHE', '000738_XSHE', '000333_XSHE', '000521_XSHE', '000555_XSHE', '000507_XSHE', '000813_XSHE',
    '000004_XSHE', '000751_XSHE', '000709_XSHE', '000027_XSHE', '000403_XSHE', '000423_XSHE', '000627_XSHE',
    '000691_XSHE', '000661_XSHE', '000410_XSHE', '000736_XSHE', '000913_XSHE', '000897_XSHE', '000422_XSHE',
    '000532_XSHE', '001226_XSHE', '000005_XSHE', '000652_XSHE', '000544_XSHE', '000611_XSHE', '000626_XSHE',
    '000581_XSHE', '000900_XSHE', '000519_XSHE', '000717_XSHE', '000886_XSHE', '000068_XSHE', '000633_XSHE',
    '000722_XSHE', '000020_XSHE', '000150_XSHE', '001202_XSHE', '000725_XSHE', '000088_XSHE', '000010_XSHE',
    '000711_XSHE', '000559_XSHE', '000612_XSHE', '000592_XSHE', '000628_XSHE', '000545_XSHE', '000504_XSHE',
    '000993_XSHE', '000530_XSHE', '000637_XSHE', '001201_XSHE', '000040_XSHE', '000759_XSHE', '000823_XSHE',
    '000338_XSHE', '000584_XSHE', '000488_XSHE', '000039_XSHE', '000702_XSHE', '000776_XSHE', '000601_XSHE',
    '000680_XSHE', '000089_XSHE', '000721_XSHE', '000698_XSHE', '000768_XSHE', '000798_XSHE', '000903_XSHE',
    '000670_XSHE', '000538_XSHE', '000513_XSHE', '000937_XSHE', '000833_XSHE', '001211_XSHE', '000673_XSHE',
    '000731_XSHE', '000852_XSHE', '000622_XSHE', '000998_XSHE', '000426_XSHE', '000587_XSHE', '000952_XSHE',
    '000863_XSHE', '000727_XSHE', '001319_XSHE', '000816_XSHE', '000830_XSHE', '000534_XSHE', '000078_XSHE',
    '000807_XSHE', '001207_XSHE', '000158_XSHE', '000930_XSHE', '000419_XSHE', '000925_XSHE', '000065_XSHE',
    '000582_XSHE', '000571_XSHE', '000066_XSHE', '000811_XSHE', '000875_XSHE', '000667_XSHE', '001270_XSHE',
    '000568_XSHE', '001217_XSHE', '000728_XSHE', '000851_XSHE', '000008_XSHE', '000997_XSHE', '000155_XSHE',
    '000856_XSHE', '000795_XSHE', '000566_XSHE', '000045_XSHE', '001316_XSHE', '000032_XSHE', '000868_XSHE',
    '000596_XSHE', '000676_XSHE', '000028_XSHE', '000523_XSHE', '000525_XSHE', '000428_XSHE', '001259_XSHE',
    '000735_XSHE', '000650_XSHE', '000678_XSHE', '000687_XSHE', '000541_XSHE', '001296_XSHE', '000557_XSHE',
    '000970_XSHE', '000061_XSHE', '000701_XSHE', '000683_XSHE', '000931_XSHE', '000668_XSHE', '000902_XSHE',
    '000796_XSHE', '000404_XSHE', '000639_XSHE', '001323_XSHE', '000505_XSHE', '000605_XSHE', '000803_XSHE',
    '001206_XSHE', '001216_XSHE', '000042_XSHE', '000657_XSHE', '001218_XSHE', '000949_XSHE', '000682_XSHE',
    '000790_XSHE', '000632_XSHE', '000876_XSHE', '000968_XSHE', '000783_XSHE', '000401_XSHE', '001212_XSHE',
    '000518_XSHE', '000797_XSHE', '000662_XSHE', '000671_XSHE', '000716_XSHE', '000898_XSHE', '000417_XSHE',
    '000737_XSHE', '000755_XSHE', '001283_XSHE', '000819_XSHE', '000880_XSHE', '001309_XSHE', '000631_XSHE',
    '000636_XSHE', '001268_XSHE', '000686_XSHE', '000533_XSHE', '000516_XSHE', '000666_XSHE', '000550_XSHE',
    '000415_XSHE', '000960_XSHE', '000935_XSHE', '000548_XSHE', '000001_XSHE', '000905_XSHE', '000906_XSHE',
    '000786_XSHE', '000719_XSHE', '000967_XSHE', '000060_XSHE', '000063_XSHE', '000809_XSHE', '000540_XSHE'
]
CommonShares471 = [
    '000768_XSHE', '000685_XSHE', '000878_XSHE', '000568_XSHE', '000520_XSHE', '000629_XSHE', '001203_XSHE',
    '000159_XSHE', '000608_XSHE', '000593_XSHE', '000750_XSHE', '000547_XSHE', '000573_XSHE', '000612_XSHE',
    '000032_XSHE', '000039_XSHE', '001207_XSHE', '000876_XSHE', '000415_XSHE', '000656_XSHE', '000785_XSHE',
    '000541_XSHE', '000422_XSHE', '000868_XSHE', '000883_XSHE', '000679_XSHE', '001317_XSHE', '000158_XSHE',
    '000820_XSHE', '000912_XSHE', '000816_XSHE', '000400_XSHE', '001288_XSHE', '000688_XSHE', '000691_XSHE',
    '000536_XSHE', '000534_XSHE', '000513_XSHE', '000728_XSHE', '000413_XSHE', '001215_XSHE', '000061_XSHE',
    '000530_XSHE', '000778_XSHE', '000791_XSHE', '000882_XSHE', '000801_XSHE', '000088_XSHE', '000980_XSHE',
    '000722_XSHE', '000029_XSHE', '000089_XSHE', '000757_XSHE', '000016_XSHE', '000528_XSHE', '000423_XSHE',
    '000913_XSHE', '000901_XSHE', '000410_XSHE', '000501_XSHE', '000825_XSHE', '000681_XSHE', '000581_XSHE',
    '000166_XSHE', '000623_XSHE', '000525_XSHE', '000639_XSHE', '000062_XSHE', '000928_XSHE', '000687_XSHE',
    '000710_XSHE', '000585_XSHE', '000921_XSHE', '000507_XSHE', '000839_XSHE', '000611_XSHE', '000048_XSHE',
    '000822_XSHE', '000830_XSHE', '000711_XSHE', '000058_XSHE', '000019_XSHE', '000563_XSHE', '000626_XSHE',
    '000430_XSHE', '000902_XSHE', '000807_XSHE', '000529_XSHE', '000906_XSHE', '000510_XSHE', '000156_XSHE',
    '000030_XSHE', '000426_XSHE', '000888_XSHE', '000852_XSHE', '000584_XSHE', '000609_XSHE', '000955_XSHE',
    '000826_XSHE', '000831_XSHE', '000586_XSHE', '000819_XSHE', '001205_XSHE', '000806_XSHE', '000650_XSHE',
    '000100_XSHE', '000555_XSHE', '000977_XSHE', '000050_XSHE', '000797_XSHE', '000553_XSHE', '000544_XSHE',
    '000909_XSHE', '000338_XSHE', '000759_XSHE', '000800_XSHE', '000923_XSHE', '000009_XSHE', '000683_XSHE',
    '000539_XSHE', '000813_XSHE', '000049_XSHE', '000962_XSHE', '000506_XSHE', '000799_XSHE', '000001_XSHE',
    '000877_XSHE', '000690_XSHE', '000960_XSHE', '000551_XSHE', '000587_XSHE', '000680_XSHE', '000006_XSHE',
    '000869_XSHE', '000953_XSHE', '000863_XSHE', '000429_XSHE', '000682_XSHE', '000559_XSHE', '000663_XSHE',
    '000926_XSHE', '000407_XSHE', '000408_XSHE', '001219_XSHE', '000403_XSHE', '000767_XSHE', '000752_XSHE',
    '000411_XSHE', '000766_XSHE', '000518_XSHE', '000060_XSHE', '000404_XSHE', '000729_XSHE', '000012_XSHE',
    '000815_XSHE', '000402_XSHE', '000652_XSHE', '000935_XSHE', '000881_XSHE', '000590_XSHE', '000592_XSHE',
    '000965_XSHE', '000705_XSHE', '000716_XSHE', '000695_XSHE', '000666_XSHE', '000732_XSHE', '000726_XSHE',
    '000978_XSHE', '000692_XSHE', '000598_XSHE', '000099_XSHE', '000823_XSHE', '000096_XSHE', '000546_XSHE',
    '000514_XSHE', '000068_XSHE', '000605_XSHE', '000002_XSHE', '000425_XSHE', '000589_XSHE', '000488_XSHE',
    '000638_XSHE', '000786_XSHE', '000850_XSHE', '000999_XSHE', '000988_XSHE', '000582_XSHE', '000886_XSHE',
    '000037_XSHE', '000973_XSHE', '000301_XSHE', '000751_XSHE', '000420_XSHE', '000719_XSHE', '000567_XSHE',
    '000597_XSHE', '000561_XSHE', '000969_XSHE', '000920_XSHE', '000042_XSHE', '000895_XSHE', '000957_XSHE',
    '000961_XSHE', '000151_XSHE', '000739_XSHE', '000715_XSHE', '000925_XSHE', '001208_XSHE', '000949_XSHE',
    '000417_XSHE', '000615_XSHE', '000004_XSHE', '000007_XSHE', '000859_XSHE', '000809_XSHE', '000981_XSHE',
    '000576_XSHE', '000531_XSHE', '000625_XSHE', '000776_XSHE', '000727_XSHE', '000498_XSHE', '000620_XSHE',
    '000828_XSHE', '000637_XSHE', '000552_XSHE', '000967_XSHE', '000919_XSHE', '000861_XSHE', '000833_XSHE',
    '000011_XSHE', '000622_XSHE', '000570_XSHE', '000982_XSHE', '000782_XSHE', '000070_XSHE', '000046_XSHE',
    '000761_XSHE', '000707_XSHE', '000922_XSHE', '000963_XSHE', '000523_XSHE', '000509_XSHE', '000572_XSHE',
    '000537_XSHE', '000045_XSHE', '000968_XSHE', '000078_XSHE', '000651_XSHE', '000903_XSHE', '001206_XSHE',
    '000700_XSHE', '000952_XSHE', '000591_XSHE', '000157_XSHE', '000595_XSHE', '000571_XSHE', '000550_XSHE',
    '000419_XSHE', '000661_XSHE', '000720_XSHE', '000708_XSHE', '000875_XSHE', '000837_XSHE', '000989_XSHE',
    '000930_XSHE', '000613_XSHE', '000958_XSHE', '000632_XSHE', '000890_XSHE', '000056_XSHE', '000540_XSHE',
    '000990_XSHE', '000938_XSHE', '000898_XSHE', '000713_XSHE', '000917_XSHE', '000333_XSHE', '000566_XSHE',
    '000035_XSHE', '000633_XSHE', '000983_XSHE', '000616_XSHE', '000673_XSHE', '000718_XSHE', '000565_XSHE',
    '000504_XSHE', '000635_XSHE', '000936_XSHE', '001211_XSHE', '000066_XSHE', '000005_XSHE', '000025_XSHE',
    '000987_XSHE', '000545_XSHE', '001267_XSHE', '001213_XSHE', '000698_XSHE', '000777_XSHE', '000887_XSHE',
    '000702_XSHE', '000905_XSHE', '000063_XSHE', '000893_XSHE', '000090_XSHE', '000996_XSHE', '000796_XSHE',
    '000701_XSHE', '000678_XSHE', '000997_XSHE', '000836_XSHE', '000059_XSHE', '000933_XSHE', '000803_XSHE',
    '000610_XSHE', '000755_XSHE', '000937_XSHE', '000810_XSHE', '000725_XSHE', '000667_XSHE', '000153_XSHE',
    '000951_XSHE', '000155_XSHE', '000697_XSHE', '000558_XSHE', '000753_XSHE', '000976_XSHE', '000835_XSHE',
    '000671_XSHE', '000017_XSHE', '000665_XSHE', '000897_XSHE', '000416_XSHE', '000792_XSHE', '000995_XSHE',
    '000929_XSHE', '000677_XSHE', '000428_XSHE', '000779_XSHE', '000026_XSHE', '000795_XSHE', '000856_XSHE',
    '000717_XSHE', '000606_XSHE', '001217_XSHE', '000519_XSHE', '000031_XSHE', '000543_XSHE', '000631_XSHE',
    '000783_XSHE', '000780_XSHE', '001212_XSHE', '001296_XSHE', '000812_XSHE', '000630_XSHE', '000737_XSHE',
    '000669_XSHE', '000959_XSHE', '000738_XSHE', '000971_XSHE', '000950_XSHE', '000858_XSHE', '000034_XSHE',
    '000838_XSHE', '000023_XSHE', '000027_XSHE', '000731_XSHE', '000655_XSHE', '001216_XSHE', '000818_XSHE',
    '000789_XSHE', '000021_XSHE', '000908_XSHE', '000931_XSHE', '000055_XSHE', '000601_XSHE', '000596_XSHE',
    '000998_XSHE', '000712_XSHE', '000790_XSHE', '000798_XSHE', '000554_XSHE', '000517_XSHE', '000860_XSHE',
    '000603_XSHE', '000628_XSHE', '000911_XSHE', '000617_XSHE', '001202_XSHE', '000636_XSHE', '000762_XSHE',
    '000505_XSHE', '000735_XSHE', '000932_XSHE', '001201_XSHE', '000736_XSHE', '000848_XSHE', '000516_XSHE',
    '000069_XSHE', '000524_XSHE', '000756_XSHE', '000010_XSHE', '000028_XSHE', '000993_XSHE', '000600_XSHE',
    '000409_XSHE', '000065_XSHE', '000972_XSHE', '000533_XSHE', '000885_XSHE', '000975_XSHE', '000557_XSHE',
    '000829_XSHE', '000721_XSHE', '000915_XSHE', '000526_XSHE', '000538_XSHE', '000703_XSHE', '000900_XSHE',
    '000607_XSHE', '000401_XSHE', '001209_XSHE', '000521_XSHE', '001210_XSHE', '000723_XSHE', '000889_XSHE',
    '000892_XSHE', '000627_XSHE', '000811_XSHE', '000548_XSHE', '000008_XSHE', '000758_XSHE', '000970_XSHE',
    '000899_XSHE', '000927_XSHE', '000036_XSHE', '000532_XSHE', '000560_XSHE', '000014_XSHE', '000619_XSHE',
    '000910_XSHE', '000599_XSHE', '001218_XSHE', '000821_XSHE', '000802_XSHE', '000918_XSHE', '000880_XSHE',
    '000038_XSHE', '000851_XSHE', '000668_XSHE', '000793_XSHE', '000709_XSHE', '000421_XSHE', '000564_XSHE',
    '000676_XSHE', '000150_XSHE', '000733_XSHE', '000020_XSHE', '000657_XSHE', '000659_XSHE', '000948_XSHE',
    '000966_XSHE', '000672_XSHE', '000788_XSHE', '000040_XSHE', '000985_XSHE', '000503_XSHE', '000862_XSHE',
    '000686_XSHE', '000502_XSHE'
]
SharesZhongZheng500 = CommonShares471

CommonDate = [
    '20210104T093100',  # 2021-01-04, 0                       0 days
    '20210406T093100',  # 2021-04-06, 0+3 months             60 days
    '20220104T093100',  # 2022-01-04, 12+0 months           243 days
    '20220406T093100',  # 2022-04-06, 12+3 months           302 days
    '20220831T150100',  # 2022-08-31, 97364, 12+8 months    404 days
    '20220913T150100',  # 2022-09-13, 99292, 12+9 months    412 days
]
K_LineCsvNames = [
    'c2c_1M.csv',
    'c2hl_1M.csv',
    'h2l_1M.csv',
    'o2hl_1M.csv',
    # 'act_volume_1M.csv',
    # 'act_volume_1M_pct.csv',
    # 'act_volume_1M_pct_scale.csv',
    'adj_volume_1M.csv',
    # 'volume_1M_scaled.csv'
    # 'amount_1M.csv',
    'amount_1M_pct.csv',
]

FactorCsvNames96 = [
    'gj_ask1_amount_avg.csv',
    'gj_ask1_amount_chg.csv',
    'gj_ask1_volume_avg.csv',
    'gj_ask1_volume_chg.csv',
    'gj_ask5_amount_avg.csv',
    'gj_ask5_amount_chg.csv',
    'gj_ask5_volume_avg.csv',
    'gj_ask5_volume_chg.csv',
    'gj_bid1_amount_avg.csv',
    'gj_bid1_amount_chg.csv',
    'gj_bid1_volume_avg.csv',
    'gj_bid1_volume_chg.csv',
    'gj_bid5_amount_avg.csv',
    'gj_bid5_amount_chg.csv',
    'gj_bid5_volume_avg.csv',
    'gj_bid5_volume_chg.csv',
    'gj_buy_bigorder_adjvolume_alladjvolume_thres0_1.csv',
    'gj_buy_bigorder_adjvolume_alladjvolume_thres1_2.csv',
    'gj_buy_bigorder_adjvolume_alladjvolume_thres2_3.csv',
    'gj_buy_bigorder_adjvolume_alladjvolume_thres3.csv',
    'gj_buy_bigorder_adjvolume_alllogadjvolume_thres0_1.csv',
    'gj_buy_bigorder_adjvolume_alllogadjvolume_thres1_2.csv',
    'gj_buy_bigorder_adjvolume_alllogadjvolume_thres2_3.csv',
    'gj_buy_bigorder_adjvolume_alllogadjvolume_thres3.csv',
    'gj_buy_bigorder_amount_alladjvolume_thres0_1.csv',
    'gj_buy_bigorder_amount_alladjvolume_thres1_2.csv',
    'gj_buy_bigorder_amount_alladjvolume_thres2_3.csv',
    'gj_buy_bigorder_amount_alladjvolume_thres3.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres0_1.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres0_1_pct.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres1_2.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres1_2_pct.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres2_3.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres2_3_pct.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres3.csv',
    'gj_buy_bigorder_amount_alllogadjvolume_thres3_pct.csv',
    'gj_buy_bigorder_ordernum_alladjvolume_thres0_1.csv',
    'gj_buy_bigorder_ordernum_alladjvolume_thres1_2.csv',
    'gj_buy_bigorder_ordernum_alladjvolume_thres2_3.csv',
    'gj_buy_bigorder_ordernum_alladjvolume_thres3.csv',
    'gj_buy_bigorder_ordernum_alllogadjvolume_thres0_1.csv',
    'gj_buy_bigorder_ordernum_alllogadjvolume_thres1_2.csv',
    'gj_buy_bigorder_ordernum_alllogadjvolume_thres2_3.csv',
    'gj_buy_bigorder_ordernum_alllogadjvolume_thres3.csv',
    'gj_buy_smallorder_adjvolume_alladjvolume_thres0.csv',
    'gj_buy_smallorder_adjvolume_alllogadjvolume_thres0.csv',
    'gj_buy_smallorder_amount_alladjvolume_thres0.csv',
    'gj_buy_smallorder_amount_alllogadjvolume_thres0.csv',
    'gj_buy_smallorder_amount_alllogadjvolume_thres0_pct.csv',
    'gj_buy_smallorder_ordernum_alladjvolume_thres0.csv',
    'gj_buy_smallorder_ordernum_alllogadjvolume_thres0.csv',
    'gj_buy_trade_ordernum.csv',
    'gj_sell_bigorder_adjvolume_alladjvolume_thres0_1.csv',
    'gj_sell_bigorder_adjvolume_alladjvolume_thres1_2.csv',
    'gj_sell_bigorder_adjvolume_alladjvolume_thres2_3.csv',
    'gj_sell_bigorder_adjvolume_alladjvolume_thres3.csv',
    'gj_sell_bigorder_adjvolume_alllogadjvolume_thres0_1.csv',
    'gj_sell_bigorder_adjvolume_alllogadjvolume_thres1_2.csv',
    'gj_sell_bigorder_adjvolume_alllogadjvolume_thres2_3.csv',
    'gj_sell_bigorder_adjvolume_alllogadjvolume_thres3.csv',
    'gj_sell_bigorder_amount_alladjvolume_thres0_1.csv',
    'gj_sell_bigorder_amount_alladjvolume_thres1_2.csv',
    'gj_sell_bigorder_amount_alladjvolume_thres2_3.csv',
    'gj_sell_bigorder_amount_alladjvolume_thres3.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres0_1.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres0_1_pct.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres1_2.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres1_2_pct.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres2_3.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres2_3_pct.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres3.csv',
    'gj_sell_bigorder_amount_alllogadjvolume_thres3_pct.csv',
    'gj_sell_bigorder_ordernum_alladjvolume_thres0_1.csv',
    'gj_sell_bigorder_ordernum_alladjvolume_thres1_2.csv',
    'gj_sell_bigorder_ordernum_alladjvolume_thres2_3.csv',
    'gj_sell_bigorder_ordernum_alladjvolume_thres3.csv',
    'gj_sell_bigorder_ordernum_alllogadjvolume_thres0_1.csv',
    'gj_sell_bigorder_ordernum_alllogadjvolume_thres1_2.csv',
    'gj_sell_bigorder_ordernum_alllogadjvolume_thres2_3.csv',
    'gj_sell_bigorder_ordernum_alllogadjvolume_thres3.csv',
    'gj_sell_smallorder_adjvolume_alladjvolume_thres0.csv',
    'gj_sell_smallorder_adjvolume_alllogadjvolume_thres0.csv',
    'gj_sell_smallorder_amount_alladjvolume_thres0.csv',
    'gj_sell_smallorder_amount_alllogadjvolume_thres0.csv',
    'gj_sell_smallorder_amount_alllogadjvolume_thres0_pct.csv',
    'gj_sell_smallorder_ordernum_alladjvolume_thres0.csv',
    'gj_sell_smallorder_ordernum_alllogadjvolume_thres0.csv',
    'gj_sell_trade_ordernum.csv',
    'gj_tradenum.csv',
    'gj_willingbuy_act_volume.csv',
    'gj_willingbuy_amount.csv',
    'gj_willingbuy_amount_pct.csv',
    'gj_willingbuy_tradenum.csv',
    'gj_willingsell_act_volume.csv',
    'gj_willingsell_amount.csv',
    'gj_willingsell_tradenum.csv',
]
FactorCsvNames19 = [
    'act_volume_by_freeShares.csv',
    'amount_by_mktValue.csv',
    'buysell_ordernum_gap.csv',
    'gj_active_buysell_l_order_amount_gap_by_DayMA10.csv',
    'gj_active_buysell_l_order_ordernum_gap_by_ordernum.csv',
    'gj_askbid1_amount_gap.csv',
    'gj_askbid5_amount_gap.csv',
    'gj_bigorder_thres2_buysell_amount_gap_by_DayMA10.csv',
    'gj_tickimpact_p1m_ret0_l_buysell_amount_gap_by_DayMA10.csv',
    'gj_tickimpact_ret0_l_buysell_amount_gap_by_DayMA10.csv',
    'gj_willing_buysell_amount_gap.csv',
    'gj_willing_buysell_amount_gap_by_DayMA10.csv',
    'gj_willing_buysell_tradenum_gap.csv',
    'gj_willing_buysell_tradenum_gap_by_DayMA10.csv',
    'gj_withdrawal_buysell_ordernum_gap_by_ordernum.csv',
    'gj_withdrawal_buysell_volume_gap_by_askbid5VolDMA10.csv',
    'log_c_2_c.csv',
    'log_c_2_c_cumsum.csv',
    'tradenum_by_DailyMA10.csv'
]
FactorCsvNames = FactorCsvNames19

DataDirPC = 'C:/Yonv/code/data/shares_data'
DataDirSever = '/home/develop/workspace/data/shares_data'
DataDir = DataDirPC if os.name == 'nt' else DataDirSever

TempDirPC = 'C:/Yonv/code/data/shares_data'
TempDirSever = '/home/develop/workspace/data/shares_temp'
TempDir = TempDirPC if os.name == 'nt' else TempDirSever


def demo__load_df(data_dir=DataDirSever):
    data_path = f"{data_dir}/c2c_1M.csv"

    # df_generator = pd.read_csv(data_path, chunksize=241 * 5, skiprows=0)
    df = pd.read_csv(data_path)
    assert len(df) % 241 == 0
    # print(df['Unnamed: 0'][0:60 * 4:60])
    """
    0      20210104T093100 20210104T103000 09:30~
    60     20210104T103100 20210104T113000      ~11:30
    120    20210104T130100 20210104T140000 13:00~
    180    20210104T140100 20210104T150000      ~15:00
    240    20210104T150100

    (2+2) * 60 + 1 = 241 min per day
    """

    # print(df)
    """
    ./c2c_1M.csv
                Unnamed: 0  000001.XSHE  000002.XSHE  000004.XSHE  ...  688819.XSHG  688981.XSHG  689009.XSHG
    0      20210104T093100    -0.010471    -0.008803    -0.001450  ...          0.0    -0.008395     0.012627
    1      20210104T093200    -0.005291    -0.002842    -0.005324  ...          0.0    -0.002246     0.002214
    2      20210104T093300     0.002128    -0.003919     0.004866  ...          0.0    -0.007792    -0.013953
    3      20210104T093400    -0.009023    -0.001073    -0.000484  ...          0.0    -0.008551    -0.004245
    4      20210104T093500    -0.005356     0.002148    -0.002907  ...          0.0     0.005105     0.009119
    ...                ...          ...          ...          ...  ...          ...          ...          ...
    99287  20220913T145700     0.000000     0.000000     0.000000  ...          0.0     0.000000     0.000000
    99288  20220913T145800     0.000000     0.000000     0.000000  ...          0.0     0.000000     0.000000
    99289  20220913T145900     0.000000     0.000000     0.000000  ...          0.0     0.000000     0.000000
    99290  20220913T150000     0.000000     0.000000     0.000000  ...          0.0     0.000000     0.000000
    99291  20220913T150100     0.000000     0.000000     0.000000  ...          0.0     0.000000     0.000000
    [99292 rows x 4908 columns] !!!!!!!!!!!!
    Date range:     2021-01-04 to 2022-09-13 !!!!!!!!!!!!!
    Date format:    year%m%dT%h%m%s

    ./gj_ask1_amount_avg.csv
                Unnamed: 0  000001.XSHE  000002.XSHE  000004.XSHE  ...  688819.XSHG  688981.XSHG  689009.XSHG
    0      20210104T093100   1437546.25    956475.20     77048.00  ...          0.0   5341503.50    913961.70
    1      20210104T093200    351602.25   1038782.30    240347.25  ...          0.0    259010.23    105255.76
    2      20210104T093300    423963.60    526395.80     47950.90  ...          0.0    213581.33    132265.16
    3      20210104T093400   3943867.75    363553.25     25708.30  ...          0.0    646228.80    168884.34
    4      20210104T093500   6302247.00    309023.34     70992.20  ...          0.0    737690.80     46302.72
    ...                ...          ...          ...          ...  ...          ...          ...          ...
    97359  20220831T145700         0.00         0.00         0.00  ...          0.0         0.00         0.00
    97360  20220831T145800         0.00         0.00         0.00  ...          0.0         0.00         0.00
    97361  20220831T145900         0.00         0.00         0.00  ...          0.0         0.00         0.00
    97362  20220831T150000         0.00         0.00         0.00  ...          0.0         0.00         0.00
    97363  20220831T150100         0.00         0.00         0.00  ...          0.0         0.00         0.00
    [97364 rows x 4893 columns] !!!!!!!!!!!
    Date range:     2021-01-04 to 2022-08-31 !!!!!!!!!!!
    Date format:    year%m%dT%h%m%s
    """
    return df


def get_common_date(data_dir=DataDirSever):
    check_mode = False

    rows = []

    names = list(os.listdir(data_dir))
    names = names[:2] if check_mode else names

    import io
    import tailer
    """
    https://stackoverflow.com/questions/17108250/efficiently-read-last-n-rows-of-csv-into-dataframe
    https://stackoverflow.com/a/45881056/9293137

    Efficiently Read last 'n' rows of CSV into DataFrame
    read large csv, last rows, pandas, DataFrame
    """
    from pandas._typing import ReadCsvBuffer

    for name in names:
        if name[-4:] != '.csv':
            continue
        path = f"{data_dir}/{name}"
        with open(path) as file:
            last_lines = tailer.tail(file, 2)

        io_string: ReadCsvBuffer = io.StringIO('\n'.join(last_lines))
        df = pd.read_csv(io_string, header=None)
        rows.append(df.iloc[-1, 0])
    rows = set(rows)
    rows = sorted(rows)
    return rows


def get_common_shares(data_dir=DataDirSever):
    check_mode = False

    cols = []

    names = list(os.listdir(data_dir))
    names = names[:2] if check_mode else names
    for name in names:
        if name[-4:] != '.csv':
            continue
        path = f"{data_dir}/{name}"
        chunk_df = pd.read_csv(path, chunksize=32)
        df = next(chunk_df)
        col = df.columns.to_list()
        cols.append(col)

    stock_sets = [set(columns) for columns in cols]
    common_cols = stock_sets[0] - {'Unnamed: 0'}
    common_cols = common_cols.intersection(*stock_sets)
    common_cols = sorted(common_cols)  # len(common_cols) == 4877
    return common_cols


def get_common_shares_from_idx_500_csv(data_dir=DataDirPC, file_name="stk_universe_idx_500.csv"):
    save_path = f"{data_dir}/{file_name}"

    if not os.path.isfile(save_path):
        import ideadata.stock.stock_data as stock_data
        df0 = stock_data.get_stk_universe(begin_date='20210101',
                                          end_date='20220929',
                                          field=["date", 'sec_id', "IDX_500"])
        df1 = df0.pivot('date', 'sec_id', 'IDX_500')
        df1.to_csv(save_path)
    else:
        df1 = pd.read_csv(save_path)

    df2 = df1[df1['date'] == '2022-01-04']
    idx_dict = df2.to_dict()
    share_names = [k for k in idx_dict.keys() if k != 'date']
    share_weights = [next(iter(d.values())) for d in idx_dict.values()]
    share_weights = np.array([float(v) for v in share_weights if isinstance(v, float)])
    share_weights = np.nan_to_num(share_weights, nan=-np.inf)
    share_weights = share_weights[share_weights > 0]
    print(f"| shares number has weights:  {len(idx_dict.keys())}")  # len(share_names) = 4883

    share_names = [share_names[i] for i in np.argsort(share_weights)]
    print(f"| shares number in data:      {len(share_names)}")  # len(share_names) = 498

    remove_names = [
        '001236.XSHE', '001229.XSHE', '001230.XSHE', '001227.XSHE', '001318.XSHE', '001308.XSHE', '001222.XSHE',
        '001266.XSHE', '000760.XSHE', '001258.XSHE', '001234.XSHE', '001289.XSHE', '001313.XSHE', '001231.XSHE',
        '001228.XSHE', '001226.XSHE', '000670.XSHE', '001319.XSHE', '001270.XSHE', '001316.XSHE', '001259.XSHE',
        '001323.XSHE', '000662.XSHE', '001283.XSHE', '001309.XSHE', '001268.XSHE', '001238.XSHE']
    remove_names = [name.replace('.', '_') for name in remove_names]
    share_names = list(set(share_names) - set(remove_names))
    print(f"| shares number after filter: {len(share_names)}")  # len(share_names) = 471

    share_names = share_names[-500:]  # top 500 (actually 471)
    return share_names


def get_idx_cons(idx_name='上证50等权'):
    idx_to_ticker = {'上证50等权': "000050", '沪深300': "000300"}
    ticker = idx_to_ticker[idx_name]

    import uqer
    _client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')

    get_idx_cons_df = getattr(uqer.DataAPI, 'IdxConsGet')
    df = get_idx_cons_df(secID=u"", ticker=ticker, isNew=u"", intoDate=u"20141231", field=u"", pandas="1")
    print(df[['secShortName', 'consID', 'consShortName']])
    return list(df['consID'])


if __name__ == '__main__':
    # print(get_common_shares_from_idx_500_csv(data_dir=DataDirPC))
    print(get_idx_cons(idx_name='上证50等权'))
    pass
