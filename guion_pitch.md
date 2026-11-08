# Guion del video pitch (2:30 aprox., máx. 3:00) — Español, 1080p 16:9

Sugerencia: cámara + screencast de las figuras 1 y 3 y del código. Voz clara, sin música.

| Tiempo | Sección | Texto para decir | Visual |
|---|---|---|---|
| 0:00–0:15 | 1. Gancho | "¿Cómo sabes que un enlace de internet rural está fallando si lo único que ves son pings lentos o sin respuesta?" | Tú en cámara; pings en pantalla |
| 0:15–0:40 | 2. Motivación e impacto | "Las redes comunitarias dependen de pocos técnicos que visitan decenas de veredas. Visitas innecesarias o fallas tardías cuestan tiempo y conectividad." | Mapa/cronograma de veredas |
| 0:40–1:10 | 3. Solución | "Modelamos la salud del enlace como un Modelo Oculto de Markov: tres estados ocultos —Óptimo, Degradado y Falla— y cuatro resultados observables del ping." | Diagrama de estados con A y B |
| 1:10–1:30 | 4. Metodología | "Implementé Forward, Viterbi y Baum-Welch desde cero en Python y simulé cuatro escenarios con 100 réplicas cada uno y semillas fijas." | Screencast del código |
| 1:30–2:05 | 5. Resultado clave | "Con parámetros conocidos, el HMM identifica el estado oculto con 89,7 % de exactitud frente a 80,5 % de un clasificador sin memoria. Pero al aprender los parámetros, solo 42 % de las inicializaciones llegó al mejor óptimo, y la exactitud varió de 8 % a 89 %." | Figuras 1 y 3 |
| 2:05–2:25 | 6. Aporte | "El aporte es un banco de pruebas reproducible que dice cuándo vale la pena un HMM: cuando los estados son persistentes y las observaciones informativas." | Tabla resumen |
| 2:25–2:40 | 7. Cierre | "Todo el código y los resultados están en el repositorio. Gracias." | Tú en cámara + enlace |

Notas: el texto hablado suma ~230 palabras (~1:30 a 1:50 a ritmo normal); puedes ampliar con la demo del código hasta ~2:30. Si pasas de 3:00, recorta la sección 2.
