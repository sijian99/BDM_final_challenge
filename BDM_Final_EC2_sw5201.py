import sys
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely
from shapely.geometry import Point
import pyspark
from pyspark.sql.types import ArrayType, FloatType, StringType
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
df1 = spark.read.csv('/tmp/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
df1 = df1.select('placekey', 'date_range_start', 'date_range_end', 'poi_cbg', 'visitor_home_cbgs').cache()
core_places_nyc = spark.read.csv('/tmp/bdm/core-places-nyc.csv', header=True, escape='"')
core_places_nyc = core_places_nyc.select('placekey', 'naics_code').cache()
core_places_nyc = core_places_nyc.filter(core_places_nyc.naics_code.startswith('4451'))
nyc_supermarkets = core_places_nyc.join(df1, (core_places_nyc.placekey ==  df1.placekey),"inner").cache()
time_nyc_supermarkets = nyc_supermarkets.withColumn('Time', F.when((F.split('date_range_start', '-')[0] == '2019') & ((F.split('date_range_start', '-')[1] == '03') | (F.split('date_range_end', '-')[1] == '03')), "2019-03")
                                          .otherwise(F.when((F.split('date_range_start', '-')[0] == '2020') & ((F.split('date_range_start', '-')[1] == '03') | (F.split('date_range_end', '-')[1] == '03')), "2020-03")
                                                    .otherwise(F.when((F.split('date_range_start', '-')[0] == '2019') & ((F.split('date_range_start', '-')[1] == '10') | (F.split('date_range_end', '-')[1] == '10')), "2019-10")
                                                              .otherwise(F.when((F.split('date_range_start', '-')[0] == '2020') & ((F.split('date_range_start', '-')[1] == '10') | (F.split('date_range_end', '-')[1] == '10')), "2020-10")
                                                              .otherwise(None))))).na.drop(subset=["Time"])
nyc_cbg_centroids = pd.read_csv('nyc_cbg_centroids.csv')
t = Transformer.from_crs(4326, 2263)
trans_array = t.transform(nyc_cbg_centroids['latitude'],nyc_cbg_centroids['longitude'])
trans_lat = dict(zip(nyc_cbg_centroids.cbg_fips, trans_array[0]))
trans_lon = dict(zip(nyc_cbg_centroids.cbg_fips, trans_array[1]))
cbg_fips = nyc_cbg_centroids.cbg_fips.tolist()
test1 = time_nyc_supermarkets.withColumn("new_part", F.regexp_replace(F.col("visitor_home_cbgs"), "[^:,0-9]", "")) \
                     .withColumn("list_part", F.split(F.col("new_part"), ",")) \
                     .withColumn("key", F.expr("transform(list_part, x -> split(x, ':')[0])")) \
                     .withColumn("value", F.expr("transform(list_part, x -> split(x, ':')[1])")) \
                     .withColumn("map", F.map_from_arrays(F.col('key'), F.col('value')))
test1 = time_nyc_supermarkets.filter("new_part != ''")
test1 = time_nyc_supermarkets.select('poi_cbg', 'Time', 'key', 'map').cache()
#ensure home in the list of NYC cbg
home_filter = F.udf(lambda target: [i for i in target if int(i) in cbg_fips], ArrayType(StringType()))
keyby_home_filter = test1.withColumn('exist_key', home_filter(test1.key))
from pyspark.sql.functions import size
keyby_home_filter = keyby_home_filter.filter(size("exist_key") > 0)
#get elem satisfy the key filter after home_filter(in nyc)
get_num = F.udf(lambda col, num: sum(((int(num.get(i))) for i in col)))
test2 = keyby_home_filter.withColumn("sum_keyvalue", get_num(keyby_home_filter.exist_key, keyby_home_filter.map))
#culculate sum distance
dist_calcultor =F.udf(lambda start, getnum, distination: sum((int(getnum.get(i)) * Point((trans_lat.get(int(i)), trans_lon.get(int(i)))).distance(Point((trans_lat.get(int(distination)), trans_lon.get(int(distination)))))/5280 for i in start)))
test2 = test2.withColumn("sum_dis", (dist_calcultor(test2.exist_key, test2.map, test2.poi_cbg)))
test2 = test2.select('poi_cbg', 'Time', 'sum_dis', 'sum_keyvalue').cache()
test2 = test2.withColumn("sum_dis",test2.sum_dis.cast('float')).withColumn("sum_keyvalue",test2.sum_keyvalue.cast('int'))
test2.groupBy('poi_cbg').pivot('Time').agg(F.sum('sum_dis') / F.sum('sum_keyvalue')).sort("poi_cbg").write.option("header",True).csv(sys.argv[1])
