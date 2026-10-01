# NVR Vision API — Documentation API

> Snapshot en entrée, détections en sortie. YOLO26 sur CPU, Go `purego` (sans CGO).
> Références : `openapi.yaml` (contrat formel), `README.md` (quickstart).

- **Base URL** : `http://localhost:8080` (configurable via `ADDR`)
- **Format** : JSON (`application/json`) en réponse, sauf corps image en entrée
- **Version API** : `1.0.0` — préfixe `/v1` pour les routes métier
- **Auth** : optionnelle. Si `API_KEY` est vide → mode ouvert (LAN). Si définie → obligatoire sur `/v1/*`

---

## 1. Sommaire des endpoints

| Méthode | Route | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | ❌ jamais | Liveness — répond dès le boot |
| `GET` | `/ready` | ❌ jamais | Readiness — `200` quand le modèle ORT + smoke test OK, sinon `503` |
| `GET` | `/v1/models` | ✅ si `API_KEY` | Liste les `.onnx` disponibles + modèle actif |
| `POST` | `/v1/detect?min_conf=0.25&classes=...` | ✅ si `API_KEY` | Détection d'objets sur un snapshot JPEG/PNG |

Toutes les autres méthodes sur ces routes → `405 Method Not Allowed` (`{error, request_id}`).

---

## 2. Authentification

Active uniquement si le serveur est démarré avec `API_KEY` non vide.

```
X-API-Key: <clé>
# ou
Authorization: Bearer <clé>
```

- Comparaison à temps constant.
- `/health` et `/ready` ne demandent **jamais** d'auth.
- Échec → `401` + `{"error": "missing or invalid API key", "request_id": "..."}`.

Exemples :

```bash
# mode ouvert (API_KEY vide)
curl localhost:8080/health

# mode protégé
curl -H "X-API-Key: $API_KEY" localhost:8080/v1/models
curl -H "Authorization: Bearer $API_KEY" -F image=@snap.jpg localhost:8080/v1/detect
```

---

## 3. Conventions transverses

### 3.1 `request_id` / `X-Request-ID`

- Chaque requête reçoit un ID hexadécimal aléatoire 16 octets (fallback timestamp si `rand` échoue).
- Retourné **dans le header** `X-Request-ID` **et dans le corps JSON** (`request_id` sur `/v1/detect` et sur toutes les erreurs).
- Repris dans les logs JSON structurés (`log/slog`) avec `method`, `path`, `status`, `duration_ms`.

### 3.2 Enveloppe d'erreur

Toutes les erreurs sont en JSON :

```json
{
  "error": "message lisible",
  "request_id": "9f3a...c1"
}
```

| Code | Signification | Quand |
|---|---|---|
| `400` | Bad request | `min_conf` invalide, `classes` inconnue, image illisible, `Content-Type` non supporté, champ `image` manquant, dimensions hors limites |
| `401` | Unauthorized | Clé API manquante/invalide (si `API_KEY` définie) |
| `405` | Method Not Allowed | Mauvaise méthode HTTP |
| `413` | Payload Too Large | Corps > `MAX_BODY_MB` (défaut 8 Mo) |
| `499` | Client Closed Request | Client a coupé la connexion pendant l'inférence |
| `500` | Internal Server Error | Échec d'inférence, panic rattrapé par middleware `recover` |
| `503` | Service Unavailable | Modèle pas encore chargé (`/ready` + `/v1/detect`) |

### 3.3 Timeouts HTTP serveur

Définis dans `cmd/server/main.go` :

| Timeout | Valeur |
|---|---|
| `ReadHeaderTimeout` | 5 s |
| `ReadTimeout` | 15 s |
| `WriteTimeout` | 60 s |
| `IdleTimeout` | 120 s |
| Timeout smoke test au boot | 2 min |

Le décodage image est borné par les dimensions (4096 px par côté, vérifiées via `image.DecodeConfig` avant le décodage complet) plutôt que par un timeout mur.

---

## 4. `GET /health`

Liveness, répond immédiatement même si le modèle charge encore.

**Requête :**

```bash
curl -i localhost:8080/health
```

**Réponse `200` :**

```json
{ "status": "ok" }
```

---

## 5. `GET /ready`

Readiness. `503` tant que le pool de sessions ORT n'est pas chargé **et** que le smoke test de démarrage n'a pas réussi.

Chaîne de boot (`cmd/server/main.go`) : `ensureModel` (download `MODEL_URL` si besoin) → `detector.New` (`dlopen` + `learnLayout` par inférence dummy zéro) → `smokeDetect` (1 vraie inférence sur `/app/smoke.jpg` ou `tests/testdata/bus.jpg`, sinon gradient synthétique) → `MarkReady()`.

**Requêtes :**

```bash
curl -i localhost:8080/health   # toujours 200
curl -i localhost:8080/ready    # 503 puis 200
```

**Réponse `200` modèle prêt :**

```json
{ "ready": true, "model": "yolo26n" }
```

**Réponse `503` modèle pas prêt :**

```json
{ "ready": false }
```

> `model` = nom court dérivé de `MODEL_PATH` (basename sans extension, suffixe `-e2e` retiré). Ex. `/models/yolo26n.onnx` → `yolo26n`.

---

## 6. `GET /v1/models`

Liste les fichiers `*.onnx` trouvés dans `modelsDir` (= dossier de `MODEL_PATH`, `/models_custom`, `./models`, `./models_custom`, dédupliqués) + le modèle actif.

**Requête :**

```bash
curl localhost:8080/v1/models
# protégé :
curl -H "X-API-Key: $API_KEY" localhost:8080/v1/models
```

**Réponse `200` :**

```json
{
  "active": "yolo26n",
  "models": [
    { "name": "yolo26n", "size_mb": 12.4 },
    { "name": "yolo26s", "size_mb": 38.1 }
  ]
}
```

Champs :

| Champ | Type | Description |
|---|---|---|
| `active` | string | Nom court du modèle chargé (`shortModelName`) |
| `models[].name` | string | Basename sans `.onnx` |
| `models[].size_mb` | number | Taille en Mio (`bytes / 2^20`) |

Les doublons (même chemin absolu) et les entrées non-fichiers sont ignorés.

---

## 7. `POST /v1/detect`

Détecte les objets d'un snapshot. NVR → JPEG/PNG, API → JSON.

### 7.1 Query params

| Param | Type | Défaut | Règles |
|---|---|---|---|
| `min_conf` | number `0..1` | `MIN_CONF` (défaut `0.25`) | `ParseFloat`, espaces trimés. Hors `0..1` ou non-numérique → `400 invalid min_conf "..." (must be 0..1)` |
| `classes` | string CSV | vide = toutes | Noms COCO insensibles à la casse, espaces trimés : `?classes=person,car,bicycle`. Label inconnu → `400 unknown class "..."`. Entrées vides ignorées ; un filtre qui ne résout aucune classe (ex. `classes=,`) → `400`. |

Exemples :

```bash
curl -F image=@snap.jpg "localhost:8080/v1/detect?min_conf=0.3&classes=person,car,bicycle"
curl -F image=@snap.jpg "localhost:8080/v1/detect"   # = MIN_CONF serveur, toutes classes
```

### 7.2 Corps de requête — 2 variantes

**A. `multipart/form-data` (recommandé pour NVR/curl) — champ unique `image` :**

```bash
curl -F image=@snap.jpg "localhost:8080/v1/detect?min_conf=0.3"
```

- `ParseMultipartForm(maxBytes)`, puis `FormFile("image")`.
- Champ manquant → `400 multipart field "image" is required`.
- Multipart invalide → `400 invalid multipart body`.
- Fichier vide → `400 empty image`.

**B. Octets bruts — `Content-Type: image/jpeg` ou `image/png` (`image/jpg` accepté) :**

```bash
curl -H "Content-Type: image/jpeg" --data-binary @snap.jpg "localhost:8080/v1/detect"
curl -H "X-API-Key: $API_KEY" -H "Content-Type: image/jpeg" --data-binary @snap.jpg "localhost:8080/v1/detect?min_conf=0.25"
```

- Tout autre `Content-Type` (ex. `application/json`) → `400 unsupported content type (use multipart field "image" or raw image/jpeg|image/png)`.
- Corps vide → `400 empty image`.

Dans les deux cas :

- Corps plafonné par `http.MaxBytesReader` à `MAX_BODY_MB` (défaut 8 Mo). Dépassement → `413 body too large`.
- Aucun fichier temporaire — tout en mémoire.
- Dimensions vérifiées avec `image.DecodeConfig` **avant** le décodage complet (garde anti-bombe de décompression) : au-delà des limites → `400`, sans jamais allouer le buffer pixels complet.
- Décodage stdlib `image.Decode` (JPEG/PNG uniquement). Échec → `400 bad image (jpeg/png required): ...`.

### 7.3 Limites image (garde anti-bombe de décompression)

Vérifiées avec `image.DecodeConfig` **avant** le décodage complet (`decodeImage`, `internal/httpapi/handlers.go`) — la borne est donc appliquée sans allouer le buffer pixels :

| Limite | Valeur | Erreur |
|---|---|---|
| Côté max | 4096 px (`maxDimension`) | `400 image dimensions WxH exceed limits (max 4096x4096)` |
| Total pixels max | `4096*4096` (`maxPixels`, ~16,7 MP / ~64 Mio RGBA) | `400 image dimensions WxH exceed limits (max 4096x4096)` |
| Dimensions nulles | rejetées | idem |

Le nombre de décodages simultanés est aussi borné (`decodeConcurrency = 2*min(POOL_SIZE,8)`, max 16) pour plafonner la mémoire.

### 7.4 Réponse `200`

```json
{
  "model": "yolo26n",
  "width": 1920,
  "height": 1080,
  "inference_ms": 23.4,
  "request_id": "9f3a...c1",
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.87,
      "box": { "x_min": 120.5, "y_min": 200.1, "x_max": 340.7, "y_max": 800.3 },
      "box_norm": { "x_min": 0.0627, "y_min": 0.1852, "x_max": 0.1774, "y_max": 0.7409 }
    }
  ]
}
```

| Champ | Type | Description |
|---|---|---|
| `model` | string | Nom court du modèle ayant inféré |
| `width` / `height` | int | Dimensions **originales** de l'image décodée (avant letterbox) |
| `inference_ms` | number | Temps `Session.Run` seul (ms, microsecondes/1000). Exclut décodage + pré/post-traitement |
| `request_id` | string | ID de la requête (aussi header `X-Request-ID`) |
| `detections` | array | Triées par `confidence` décroissante. `[]` (jamais `null`) si rien détecté. Cap 300 |
| `detections[].class_id` | int | ID COCO `0..79` |
| `detections[].label` | string | Nom COCO (ex. `person`) |
| `detections[].confidence` | number `0..1` | Score filtré par `min_conf` |
| `detections[].box` | object | Pixels espace image originale, `x_max > x_min`, `y_max > y_min`, clampés aux bords |
| `detections[].box_norm` | object | Mêmes coords normalisées `0..1` (`x / width`, `y / height`) |

### 7.5 Pipeline d'inférence (pour interpréter les résultats)

`internal/detector/detector.go` :

1. **Pré-traitement (Go pur)** : letterbox → `640x640` (ratio conservé, pad gris `114`), `float32` `0..1`, NCHW `[1,3,640,640]` (redimensionnement `BiLinear` via `golang.org/x/image/draw`).
2. **Session ORT** : une session exclusive empruntée au pool (`SessionPool`, taille `POOL_SIZE`, défaut `NumCPU`). Sessions **non** partagées entre goroutines.
3. **Layout auto-détecté** au démarrage par inférence dummy zéro :
   - e2e NMS-free `(1, N, 6)` = `[x1,y1,x2,y2,score,class_id]` → simple filtre score.
   - raw `(1, 4+nc, anchors)` ex. `(1,84,8400)` COCO → meilleur score par ancre + NMS gloutonne **par classe** (IoU `0.45`, plancher `0.01`, cap 300).
   - raw-transposé `(1, anchors, 4+nc)` ex. `(1,8400,84)` → même décodage + NMS que raw.
4. **Post-traitement** : filtre `min_conf` (les valeurs `< 0.01` sont remontées à `0.01` pour borner le coût NMS en mode raw / raw-transposé) → filtre `classes` → suppression des `class_id` hors `0..79` → un-letterbox vers pixels originaux + clamp → suppression des boxes dégénérées → tri confiance décroissante → cap 300.

Tailles/perf indicatives (CPU) : `n` FP32 ~10–15 Mo / ~23–39 ms, `s` FP32 ~35–45 Mo / ~60–87 ms.

### 7.6 Erreurs spécifiques à `/v1/detect`

| Cas | Code | Corps |
|---|---|---|
| Modèle pas prêt (`ready=false` ou détecteur nil) | `503` | `{"error":"model not ready",...}` |
| Pool d'inférence saturé (attente > 30 s) | `503` + `Retry-After: 5` | `{"error":"server busy, try again",...}` |
| `min_conf=2` / `abc` | `400` | `{"error":"invalid min_conf \"2\" (must be 0..1)",...}` |
| `classes=nope` | `400` | `{"error":"unknown class \"nope\"",...}` |
| Octets non-image | `400` | `{"error":"bad image (jpeg/png required): ...",...}` |
| Mauvais `Content-Type` | `400` | `{"error":"unsupported content type ...",...}` |
| Champ multipart ≠ `image` | `400` | `{"error":"multipart field \"image\" is required",...}` |
| Corps > `MAX_BODY_MB` | `413` | `{"error":"body too large",...}` |
| Client déconnecté pendant `Detect` | `499` | `{"error":"client closed request",...}` |
| Échec ORT interne | `500` | `{"error":"inference failed",...}` (détail en log serveur uniquement) |

**Notes de comportement :**

- **Annulation client (499)** : le contexte interrompt l'attente d'une session du pool, mais **pas** une inférence ORT déjà lancée (le binding ne supporte pas la terminaison en cours de run). Un client qui coupe la connexion fait donc revenir la requête, mais l'inférence en cours se termine avant d'être libérée.
- **Labels** : la table COCO-80 est câblée en dur (`internal/yolo/coco.go`). Un modèle custom avec un autre nombre de classes verra ses `class_id ≥ 80` écartés et des labels potentiellement faux (un avertissement est journalisé au démarrage si le modèle n'a pas 80 classes).
- **Valeurs non finies** : les scores/boîtes `NaN`/`Inf` éventuels du modèle sont filtrés ; une valeur non finie en sortie ne peut pas produire un corps JSON invalide.

---

## 8. Classes COCO (80) — valeurs valides pour `?classes=`

`internal/yolo/coco.go`. Insensibles à la casse dans le filtre.

| ID | Label | ID | Label | ID | Label | ID | Label |
|---|---|---|---|---|---|---|---|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

---

## 9. Exemples complets

### curl (NVR)

```bash
# health / ready
curl localhost:8080/health
curl localhost:8080/ready

# detect multipart (mode ouvert)
curl -F image=@snap.jpg "localhost:8080/v1/detect?min_conf=0.3&classes=person,car,bicycle"

# detect raw bytes + clé API
curl -H "X-API-Key: $API_KEY" -H "Content-Type: image/jpeg" \
  --data-binary @snap.jpg "localhost:8080/v1/detect?min_conf=0.25"

# verbose : voir X-Request-ID
curl -i -F image=@tests/testdata/bus.jpg "localhost:8080/v1/detect?min_conf=0.3"
```

### Python (requests)

```python
import requests

# multipart
with open("snap.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8080/v1/detect",
        params={"min_conf": 0.3, "classes": "person,car"},
        files={"image": ("snap.jpg", f, "image/jpeg")},
        headers={"X-API-Key": "secret"},  # si API_KEY définie
        timeout=30,
    )
print(r.json())

# raw bytes
with open("snap.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8080/v1/detect?min_conf=0.25",
        data=f.read(),
        headers={"Content-Type": "image/jpeg"},
        timeout=30,
    )
print(r.json())
```

### Go (net/http)

```go
f, _ := os.Open("snap.jpg")
defer f.Close()
req, _ := http.NewRequest("POST", "http://localhost:8080/v1/detect?min_conf=0.3", f)
req.Header.Set("Content-Type", "image/jpeg")
req.Header.Set("X-API-Key", os.Getenv("API_KEY"))
resp, _ := http.DefaultClient.Do(req)
defer resp.Body.Close()
```

---

## 10. Configuration (variables d'environnement)

`internal/config/config.go`, `docker-compose.yml`.

| Var | Défaut (code) | Défaut compose | Contraintes | Effet API |
|---|---|---|---|---|
| `ADDR` | `:8080` | `:8080` | — | écoute HTTP |
| `MODEL_PATH` | `/models/yolo26n.onnx` | `/models/yolo26s.onnx` | fichier doit exister (ou `MODEL_URL`) | `model`, `/v1/models:active`, `/ready:model` |
| `MODEL_URL` | vide | vide | — | download lazy si fichier manquant (timeout 15 min) |
| `ORT_LIB_PATH` | `/usr/lib/libonnxruntime.so.1` | (image) | `.so` musl via apk | `dlopen` ; échec → `/ready` reste `503` |
| `IMG_SIZE` | `640` | `640` | `1..2048`, doit matcher l'export | taille letterbox ; mismatch → erreur dummy inference au boot |
| `MIN_CONF` | `0.25` | `0.25` | `0..1` | défaut de `?min_conf=` |
| `INTRA_THREADS` | `0` (=auto) | `4` | — | threads intra-op ORT par session |
| `POOL_SIZE` | `0` (=NumCPU) | `1` | `1..64` | sessions ORT concurrentes ; tuning `POOL_SIZE × INTRA_THREADS ≈ NumCPU` |
| `API_KEY` | vide (ouvert) | vide | — | si définie, exigée sur `/v1/*` |
| `MAX_BODY_MB` | `8` | `8` | `1..64` | seuil `413` |
| `LOG_LEVEL` | `info` | `debug` | `debug/info/warn/error` | logs JSON `slog` |

---

## 11. Limites & notes d'exploitation

- **CPU `amd64` uniquement**, Docker Alpine-edge (ORT musl depuis apk, pas de shim glibc). `CGO_ENABLED=0`.
- **Modèles** : `yolo26n` baked dans l'image (`/models/`), `yolo26s` via mount `./models_custom:/models_custom:ro` + `MODEL_PATH=/models_custom/yolo26s.onnx`. Ne jamais monter par-dessus `/models` (masquerait le modèle baked).
- **Concurrence** : pool de sessions ORT (`POOL_SIZE`) ; une requête = une session exclusive. Requête annulée → `499`.
- **Pas de fichiers temp** : tout en mémoire, borné par `MAX_BODY_MB` + garde pixels.
- **Santé Docker** : `HEALTHCHECK wget http://127.0.0.1:8080/health | grep '"status":"ok"'`.
- **Non-objectifs v1** : pas de tracking/ReID, pas d'ingest vidéo, pas de GPU/CUDA, pas d'INT8 par défaut, pas de modèle `s` baked.

---

## 12. Référence OpenAPI

Le contrat machine est dans `openapi.yaml` (`security: ApiKey`, `servers: [{url: http://localhost:8080}]`). Cette page `API.md` en est la version lisible et ajoute les comportements implémentés (timeouts, limites pixels, codes `405/499/500`, smoke test, pool, NMS) qui ne figurent pas dans le YAML.
