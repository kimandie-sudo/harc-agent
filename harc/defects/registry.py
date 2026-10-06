from .bowing import Bowing
from .taper import Taper
from .necking import Necking
from .undercut import Undercut
from .sidewall_angle import SidewallAngle
from .etch_stop import EtchStop
from .mask_erosion import MaskErosion
from .cd_deviation import CdDeviation
from .depth_loss import DepthLoss
from .low_aspect_ratio import LowAspectRatio

# 결함 추가 = 모듈 작성 후 여기에 등록.
# bowing↔undercut, taper↔sidewall_angle 은 같은 계열이지만 지표 단위가 달라 따로 둔다.
ALL = [Bowing(), Taper(), Necking(), Undercut(), SidewallAngle(), EtchStop(),
       MaskErosion(), CdDeviation(), DepthLoss(), LowAspectRatio()]
