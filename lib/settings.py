# Distance target
TARGET_DISTANCE_M = 20_000_000         # 20000 km

# Autonomy limits
MAX_GAS_L = 100.0
MIN_ELECTRIC_SOC = 0.20

# Timestep
DT_S = 1.0                          # 1 second resolution

# Inverter parameters
INVERTER_RATED_KW      = 220.0
INVERTER_REGEN_FACTOR = 0.60

# ICE parameters
ICE_MAX_POWER_KW       = 55.0
ICE_OPTIMUM_POWER_KW   = 38.5       # 70% of max — BSFC sweet spot
ICE_PEAK_EFFICIENCY        = 0.36

# Battery parameters
BATTERY_CAPACITY_KWH   = 85.0
INITIAL_SOC            = 0.95       # Start nearly full
INTERNAL_RESISTANCE_OHM = 0.080     # Ohmic losses at high power

# Motor Parameters
MOTOR_CONT_POWER_KW    = 150.0      # Continuous rating
MOTOR_PEAK_POWER_KW    = 220.0      # 25 s burst
PEAK_DURATION_S       = 25.0       # Max duration of peak power
MOTOR_PEAK_EFFICIENCY       = 0.965      # Efficiency at peak power

# Car parameters
VEHICLE_MASS_KG        = 1950.0
DRAG_COEFFICIENT       = 0.24
FRONTAL_AREA_M2        = 2.3
ROLLING_RESISTANCE    = 0.012

# Car initial conditions
INITIAL_SPEED_KMH = 0.0
ROAD_TYPE = 'CITY'  # Options: 'CITY', 'HIGHWAY', 'MIXED'

# EMS thresholds
REX_TRIGGER_SOC  = 0.21             # ICE starts at 30% SoC
REX_CUTOFF_SOC   = 0.95             # ICE stops at 50% SoC

# Downsampling for telemetry storage (every N seconds)
LOG_INTERVAL_S = 10