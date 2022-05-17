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
from pyspark.sql.functions import size
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
df1 = spark.read.csv('/tmp/bdm/weekly-patterns-nyc-2019-2020-sample.csv', header=True, escape='"')
df1 = df1.select('placekey', 'date_range_start', 'date_range_end', 'poi_cbg', 'visitor_home_cbgs').cache()
nyc_supermarkets = spark.read.csv('nyc_supermarkets.csv', header=True, escape='"')
#filter by nyc
nyc_supermarkets = nyc_supermarkets.select('safegraph_placekey')
get_nyc_supermarkets = nyc_supermarkets.join(df1, (nyc_supermarkets.safegraph_placekey ==  df1.placekey),"inner").cache()
#filter by time
time_nyc_supermarkets = get_nyc_supermarkets.withColumn('Time', F.when((F.split('date_range_start', '-')[0] == '2019') & ((F.split('date_range_start', '-')[1] == '03') | (F.split('date_range_end', '-')[1] == '03')), "2019-03")
                                          .otherwise(F.when((F.split('date_range_start', '-')[0] == '2020') & ((F.split('date_range_start', '-')[1] == '03') | (F.split('date_range_end', '-')[1] == '03')), "2020-03")
                                                    .otherwise(F.when((F.split('date_range_start', '-')[0] == '2019') & ((F.split('date_range_start', '-')[1] == '10') | (F.split('date_range_end', '-')[1] == '10')), "2019-10")
                                                              .otherwise(F.when((F.split('date_range_start', '-')[0] == '2020') & ((F.split('date_range_start', '-')[1] == '10') | (F.split('date_range_end', '-')[1] == '10')), "2020-10")
                                                              .otherwise(None))))).na.drop(subset=["Time"])
nyc_cbg_centroids = pd.read_csv('nyc_cbg_centroids.csv')
#transform corrdination
t = Transformer.from_crs(4326, 2263)
trans_array = t.transform(nyc_cbg_centroids['latitude'],nyc_cbg_centroids['longitude'])
trans_lat = dict(zip(nyc_cbg_centroids.cbg_fips, trans_array[0]))
trans_lon = dict(zip(nyc_cbg_centroids.cbg_fips, trans_array[1]))
cbg_fips = nyc_cbg_centroids.cbg_fips.tolist()
#get key, value, and dict
test1 = time_nyc_supermarkets.withColumn("new_part", F.regexp_replace(F.col("visitor_home_cbgs"), "[^:,0-9]", "")) .withColumn("list_part", F.split(F.col("new_part"), ",")) \
                     .withColumn("key", F.expr("transform(list_part, x -> split(x, ':')[0])")).withColumn("value", F.expr("transform(list_part, x -> split(x, ':')[1])")) \
                     .withColumn("map", F.map_from_arrays(F.col('key'), F.col('value')))
test1 = test1.filter("new_part != ''")
test1 = test1.select('poi_cbg', 'Time', 'key', 'map').cache()
#ensure home in the list of NYC cbg
home_filter = F.udf(lambda target: [i for i in target if int(i) in cbg_fips], ArrayType(StringType()))
keyby_home_filter = test1.withColumn('exist_key', home_filter(test1.key))
keyby_home_filter = keyby_home_filter.filter(size("exist_key") > 0)
#get elem satisfy the key filter after home_filter(in nyc)
get_num = F.udf(lambda col, num: sum(((int(num.get(i))) for i in col)))
test2 = keyby_home_filter.withColumn("sum_keyvalue", get_num(keyby_home_filter.exist_key, keyby_home_filter.map))
def get_list(tuples_list, target, count_map):
    list_1 = []
    for elem in tuples_list:
      list_1.extend([Point(trans_lat.get(int(elem)), trans_lon.get(int(elem))).distance(Point(trans_lat.get(int(target)), trans_lon.get(int(target))))/5280] * int(count_map.get(elem)))
    return list_1
create_list = F.udf(get_list, ArrayType(FloatType()))
new_list = test2.withColumn('new_list', create_list(test2.exist_key, test2.poi_cbg, test2.map)).cache()
pivot1 = new_list.groupBy('poi_cbg').pivot('Time').agg(F.flatten(F.collect_list("new_list"))).cache()
def get_median(list_1):
      if (len(list_1) == 0):  
       return 'Null'
      else: 
       return sorted(list_1)[int(len(list_1)/2)] 
median_f = F.udf(get_median)
pivot1.withColumn('2019-03', median_f(F.col('2019-03'))).withColumn('2019-10', median_f(F.col('2019-10'))).withColumn('2020-03', median_f(F.col('2020-03'))).withColumn('2020-10', median_f(F.col('2020-10'))).write.option("header",True).csv(sys.argv[1])