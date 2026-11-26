# Práctica: Segmentación de Clientes con K-Means y PCA en EC2 XD

## Conclusión de Resultados a los que llegue.

A partir de los resultados obtenidos, se concluye que el valor óptimo de segmentación es $K=4$, ya que logra maximizar simultáneamente el Coeficiente de Silueta (`0.6863`) y el Índice Calinski-Harabasz (`4846.2292`),
a la vez que minimiza el Índice Davies-Bouldin (`0.4625`), demostrando una alta cohesión interna y una clara separación entre los grupos de clientes sin fragmentar innecesariamente los perfiles
(como ocurre al evaluar $K=5$, donde la silueta cae a `0.5854` y Davies-Bouldin sube a `0.8195`). Asimismo, al aplicar PCA para reducir las 6 variables originales a 2 dimensiones principales, el modelo logró retener
el 87.63% de la varianza explicada, conservando casi la totalidad de la estructura de los datos para su representación visual. Finalmente, en un entorno de producción —ya sea mediante una API REST o descargando los
artefactos desde un bucket de AWS S3—, los tres archivos `.pkl` operan de manera secuencial ante cada nuevo registro: primero `scaler_clientes.pkl` normaliza las variables entrantes con los parámetros del entrenamiento,
posteriormente `modelo_kmeans_clientes.pkl` asigna el segmento correspondiente, y por último `pca_clientes.pkl` proyecta el dato en coordenadas 2D para su visualización en tiempo real.

Para que vean que es 100% real no fake + link + mega + crack agrego a este repositorio una captura de pantalla con el nombre de practica_segmentacion_ec2_evidencia.png y los archivos pkl que genera esta práctica.

XDXDXDXDDDDDDDDDDD
