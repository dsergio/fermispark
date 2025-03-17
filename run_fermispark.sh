$SPARK_HOME/bin/spark-submit --master spark://$SPARK_MASTER_IP/:$SPARK_MASTER_PORT \
    --deploy-mode client \
    --conf spark.executor.memory=5g \
    --conf spark.driver.memory=5g \
    --conf spark.driver.maxResultSize=4g \
    src/fermispark.py
