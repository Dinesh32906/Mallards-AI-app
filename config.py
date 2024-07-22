import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

SNOWFLAKE_ACCOUNT = 'nxb36304'
SNOWFLAKE_USER = 'dinesh1985'
SNOWFLAKE_PASSWORD = 'Kishan1985@'
SNOWFLAKE_ROLE = 'SYSADMIN'
SNOWFLAKE_WAREHOUSE = 'COMPUTE_WH'
SNOWFLAKE_DATABASE = 'CC_QUICKSTART_CORTEX_DOCS'
SNOWFLAKE_SCHEMA = 'DATA'
