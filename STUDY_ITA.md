# Guida Completa a ds4.c - DeepSeek V4 Flash Inference Engine

## Prefazione

Questo documento nasce come materiale di studio per sviluppatori web (come me) che vogliono capire il progetto ds4.c di antirez. Non sono un esperto di machine learning o sistemi embedded - sono un web developer che ha deciso di capire come funziona l'inferenza locale di modelli linguistici.

L'obiettivo è arrivare a comprendere il codice sorgente al livello di antirez, che significa:
- Capire ogni componente dell'architettura
- Sapere leggere e modificare il codice
- Poter contribuire al progetto
- Poter discutere tecnicamente con altri sviluppatori

Questo è un viaggio di apprendimento. Partiamo.

---

# Parte 1: Cos'è ds4.c e Perché Esiste

## 1.1 Il Contesto

Nel 2024-2025, i modelli linguistici di grandi dimensioni (LLM) sono diventati mainstream. Ma eseguire questi modelli localmente (senza usare API cloud come OpenAI, Anthropic, ecc.) è stato a lungo un problema irrisolto per utenti normali.

Vari progetti hanno cercato di risolvere questo problema:
- **llama.cpp**: il pioniere, permette di far girare modelli su CPU (e poi GPU)
- **llamafile**: versione portable di llama.cpp
- **Ollama**: wrapper user-friendly
- **vLLM**: ottimizzato per server
- **exo**: progetto decentralizzato

Ma **ds4.c** è diverso.

## 1.2 Cos'è ds4.c

**ds4.c** è un motore di inferenza (inference engine) specificamente scritto per il modello **DeepSeek V4 Flash**.

Non è un runner GGUF generico - è scritto apposta per questo modello. Questo approccio permette ottimizzazioni profonde che non sarebbero possibili in un runner generico.

Dalla README:
> "ds4.c is a small native inference engine for DeepSeek V4 Flash. It is intentionally narrow: not a generic GGUF runner, not a wrapper around another runtime, and not a framework."

## 1.3 Perché DeepSeek V4 Flash?

DeepSeek V4 Flash è un modello particolare:

### Caratteristiche distintive
- **284 miliardi di parametri totali**
- **~36 miliardi attivi per token** (grazie a MoE)
- **1 milione di token di context window** (enorme!)
- **81GB compresso** (IQ2 quantization)
- **MQTT-like tool calling** integrato

Il modello è stato sviluppato da DeepSeek (azienda cinese) e rilasciato con licenza permissive. Antirez ha deciso di creare un motore custom perché:
1. Il modello ha caratteristiche uniche (KV compression, MoE)
2. I runner generici non sfruttano queste ottimizzazioni
3. Wanted to show what's possible with proper engineering

## 1.4 Cosa lo rende speciale

### KV Cache su Disco
L'innovazione principale di ds4.c: la KV cache (la memoria del modello) viene salvata su disco, non in RAM.

Perché? Con 1M token di context, la KV cache è enorme. Anche compressa, non starebbe in RAM. Ma gli SSD moderni sono veloci abbastanza da renderla pratica.

### Quantizzazione Asimmetrica
Il modello usa una quantizzazione particolare: solo gli esperti MoE sono quantizzati in IQ2_XXS (2-bit), mentre le altre parti restano a precisione più alta. Questo mantiene la qualità pur riducendo le dimensioni.

### Specifico, non generico
Antirez ha scelto di supportare un solo modello ben fatto, piuttosto che tanti modelli mal fatti. Questo approccio permette:
- Ottimizzazioni profonde
- Testing rigoroso
- Validazione contro output ufficiali

---

# Parte 2: DeepSeek V4 Flash - Il Modello

## 2.1 Architettura MoE

MoE = **Mixture of Experts**. È una tecnica per rendere i modelli più efficienti.

### Come funziona tradizionalmente
Un modello 284B ha 284 miliardi di parametri. Ogni token viene processato da tutti questi parametri.

### Come funziona MoE
Il modello ha **64 esperti** (group of experts), ma per ogni token ne attiva solo **2** (plus shared experts).

```
Total parameters: 284B
Active per token: ~36B (solo 2 esperti + shared)
```

Questo significa:
- Velocità simile a modello 36B (non 284B)
- Memoria necessaria simile a 36B
- Ma la "conoscenza" è di 284B

### Routing
Un "router" decide quali 2 esperti attivare per ogni token. È come un sistema di routing intelligente che decide chi gestisce ogni richiesta.

## 2.2 Context Window 1M

Il modello supporta **1 milione di token** di context. Per confronto:
- GPT-4: 128k
- Claude: 200k
- Questo modello: 1M!

Questo è utile per:
- Analisi di documenti enormi
- Conversazioni lunghissime
- Raggiamento su codebase interi

## 2.3 Quantizzazione IQ2

Il modello è compresso usando **IQ2_XXS**, un formato di quantizzazione 2-bit sviluppato da llama.cpp.

### Perché 2-bit funziona
La quantizzazione tradizionale (come 8-bit) è lineare - ogni peso viene scalato linearmente.

IQ2 è diverso:
- Usa **lookup tables** per valori non lineari
- Mantiene più informazione a parità di bit
- Complesso da implementare ma funziona bene

### Layout asimmetrico
Non tutti i pesi sono quantizzati uguale:
- **MoE experts**: IQ2_XXS (2-bit) - la maggioranza
- **Down projection**: Q2_K (2-bit un po' diverso)
- **Embedding/Other**: F16 (16-bit) - per mantenere qualità

Questo "asymmetric quantization" è la chiave per mantenere la qualità.

## 2.4 Parametri del modello

Dalla config del modello:
- `block_count`: 43 layers
- `num_experts_used`: 8
- `num_experts`: 64
- `head_count`: 8
- `head_dim`: 128
- `embedding_size`: 6144
- `vocab_size`: 128000

---

# Parte 3: Concetti Tecnici Fondamentali

## 3.1 Cos'è l'inferenza

**Inferenza** = far generare testo al modello. Il opposto di "training" (addestramento).

Quando scrivi "Ciao" e il modello risponde "Ciao! Come stai?", quello è inferenza.

### Pipeline base
```
Input (prompt)
    ↓
Tokenizzazione (testo → numeri)
    ↓
Embedding (numeri → vettori)
    ↓
Forward Pass (elaborazione attraverso layers)
    ↓
Logits (output grezzi)
    ↓
Sampling (scelta prossimo token)
    ↓
Output (token scelto)
```

## 3.2 Tokenizzazione

Il modello non capisce testo - capisce solo numeri. La **tokenizzazione** converte testo in token IDs.

Esempio:
- Testo: "Ciao come stai?"
- Tokens: [1234, 5678, 9012, 3456, 7890]

Il vocabolo del modello ha 128000 token possibili.

## 3.3 Embedding

Dopo la tokenizzazione, ogni token viene convertito in un vettore di numeri (embedding).

Per DeepSeek:
- Ogni token → vettore di 6144 numeri (float)

L'embedding è una matrice: `vocab_size × embedding_size`
- 128000 × 6144 = enormissima!

In IQ2 compresso: ~1.5GB invece di ~3GB.

## 3.4 Attention (Self-Attention)

L'attention è il meccanismo centrale dei transformer.

### Come funziona
Per ogni token, il modello decide "a quali altri token prestare attenzione".

Matematicamente:
```
Output = attention(Q, K, V)

dove:
Q = Query (cosa cerco)
K = Key (cosa offro)
V = Value (il valore che offro)
```

### Self-attention
Ogni token ha Q, K, V calcolati da se stesso. "Self" perché si relaziona a sé stesso.

### Multi-head
Il modello usa 8 "heads" - 8 diverse rappresentazioni dell'attenzione. Ogni head impara pattern diversi.

### KV Cache
Durante la generazione, K e V di tutti i token precedenti vengono memorizzati nella KV cache. Così non devono essere ricalcolati.

## 3.5 Transformer Layers

Ogni layer consist di:
1. **Self-Attention** - relazioni tra token
2. **Feed-Forward** - elaborazione non-lineare
3. **Normalization** - stabilità numerica
4. **Residual connections** - gradient flow

DeepSeek ha 43 layer di questo tipo.

## 3.6 Feed-Forward con MoE

In un transformer normale, il feed-forward è:
```
Dense layer → activation → Dense layer
```

In DeepSeek, il feed-forward è MoE:
```
Input
    ↓
[Router sceglie 2 esperti]
    ↓
Esperti attivati (parallel)
    ↓
Output = weighted sum degli esperti
```

Ogni esperto è un feed-forward network separato.

## 3.7 GGUF - Il Formato File

**GGUF** = GGML Unified Format. Sviluppato per llama.cpp, ora standard de facto.

### Struttura GGUF
```
┌────────────────────────────┐
│ Header (magic, version)    │
├────────────────────────────┤
│ Metadata (key-value pairs) │
├────────────────────────────┤
│ Tensor Info Section         │
│ (nome, shape, tipo)        │
├────────────────────────────┤
│ Tensor Data Section        │
│ (i pesi compressi)          │
└────────────────────────────┘
```

### Tipi di tensori in GGUF
- `0` = F32 (float32)
- `1` = F16 (float16)
- `8` = Q8_0 (8-bit quantization)
- `10` = Q2_K (2-bit quantization)
- `16` = IQ2_XXS (2-bit con lookup tables)

## 3.8 Sampling

Dopo il forward pass, abbiamo i **logits** - un array di 128000 numeri (un per ogni possibile prossimo token).

Il sampling decide quale token scegliere:

### Greedy
Scegli il token con probabilità più alta. Facile, veloce, ma può essere ripetitivo.

### Temperature
Divide i logits per una temperatura:
- Alta (1.0+) = più casuale, creativa
- Bassa (0.1-0.5) = più deterministica, focalizzata

### Top-K, Top-P
Limita la scelta ai K token più probabili, o a quelli che sommano a P probabilità.

---

# Parte 4: Architettura di ds4.c

## 4.1 Struttura del Progetto

```
ds4/
├── ds4.c          # Motore principale (~16k righe)
├── ds4.h          # Header con defines
├── ds4_cli.c      # CLI interface
├── ds4_server.c   # Server HTTP
├── Makefile       # Build system
├── gguf/          # Directory modelli
│   └── *.gguf
├── tests/         # Test suite
├── metal/         # Kernel Metal (macOS)
└── docs/          # Documentazione
```

## 4.2 Flusso di Esecuzione

```
┌──────────────────────────────────────────────────────────┐
│                     main() in ds4_cli.c                  │
└─────────────────────────────┬────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────┐
│              ds4_engine_open() in ds4.c                  │
│  1. model_open() - carica GGUF                           │
│  2. vocab_load() - carica vocabolario                    │
│  3. config_validate_model() - valida metadata            │
│  4. weights_bind() - collega pesi al modello             │
│  5. backend init (Metal/CPU/HIP)                          │
└─────────────────────────────┬────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────┐
│              ds4_engine_chat() in ds4.c                  │
│  1. Prompt preparation                                    │
│  2. Tokenization                                          │
│  3. Generation loop                                       │
│     - prefill (una volta)                                  │
│     - decode (per ogni token)                             │
│     - sampling                                            │
└─────────────────────────────┬────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────┐
│              ds4_engine_close()                          │
│  - Free weights                                           │
│  - Free vocab                                             │
│  - Close model file                                       │
│  - Cleanup backend                                        │
└──────────────────────────────────────────────────────────┘
```

## 4.3 Strutture Dati Principali

### ds4_model
Contiene il modello GGUF caricato:
```c
typedef struct {
    int fd;
    const uint8_t *map;       // mmap del file
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    ds4_kv *kv;               // metadata key-value
    ds4_tensor *tensors;      // array di tensori
} ds4_model;
```

### ds4_weights
Puntatori ai tensori dei pesi:
```c
typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output;
    ds4_layer_weights layer[43];  // 43 layers
    // ... altri pesi
} ds4_weights;
```

### ds4_layer_weights
Pesi per un singolo layer:
```c
typedef struct {
    ds4_tensor *attn_q_a;     // Q projection (MoE)
    ds4_tensor *attn_q_b;     
    ds4_tensor *attn_kv;      // KV attention
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *ffn_gate_exps;    // MoE experts
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    // ... altri tensori
} ds4_layer_weights;
```

### ds4_tensor
Un tensore nel modello:
```c
typedef struct {
    char name[64];
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;            // 0=F32, 1=F16, 16=IQ2_XXS, etc.
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;
```

## 4.4 Backend

### Metal (macOS)
Il backend principale. Usa Metal Performance Shaders per accelerazione GPU.

### CPU
Solo per debugging. Non ottimizzato.

### HIP/ROCm (AMD GPU)
Sperimentale (il mio fork). Usa rocBLAS per matmul.

## 4.5 KV Cache

### Compressione
La KV cache è compressa:
- Raw: sliding window con head dimension
- Compressed: ratio-4 per long-term

### Su Disco
La KV viene serializzata su disco nel formato KVC.

---

# Parte 5: Come Leggere il Codice

## 5.1 Punti d'ingresso

### ds4_cli.c
```c
int main(int argc, char **argv) {
    // Parse argomenti
    // Crea engine
    // Loop chat/interactive
    // Cleanup
}
```

### ds4_engine_open()
Punto dove viene caricato tutto.

### ds4_engine_chat()
Punto dove avviene la generazione.

## 5.2 Funzioni Importanti

### weights_bind()
Collega i tensori GGUF alle strutture ds4_weights. Qui ogni tensore viene assegnato al suo ruolo (attn_q, ffn, etc.).

### matvec_any()
Moltiplicazione matrice-vettore per qualsiasi tipo di tensore. Dispatch basato sul tipo (F32, F16, Q8_0, IQ2).

### prefill_layer_major_cpu()
Prefill completo del prompt - processa tutti i token del prompt in parallelo.

### forward_token_raw_swa_cpu_decode_scratch()
Decode di un singolo token.

### ds4_vec_dot_iq2_xxs_q8_K()
Dot product per IQ2_XXS - la funzione chiave per i pesi MoE quantizzati.

## 5.3 Patterns Riconoscibili

### Error handling
```c
if (!condizione) ds4_die("messaggio");
```

### Allocazione
```c
void *xmalloc(size_t size);
void *xcalloc(size_t nmemb, size_t size);
```

### Loop su layer
```c
for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
    // process layer
}
```

---

# Parte 6: Glossario Esteso

## Termini Generali

| Termine | Significato |
|---------|-------------|
| **Inference** | Processo di generazione testo da un modello |
| **Training** | Processo di addestramento del modello |
| **Forward pass** | Una passata attraverso la rete neurale |
| **Backward pass** | Calcolo gradienti (solo training) |
| **Epoch** | Una passata completa su tutto il dataset |

## Termini Modello

| Termine | Significato |
|---------|-------------|
| **Parameters** | I numeri (pesi) che definiscono il modello |
| **Active parameters** | Quelli usati per ogni token (in MoE) |
| **Hidden size** | Dimensione dei vettori intermedi |
| **Attention heads** | Numero di teste di attenzione |
| **Layer count** | Numero di transformer layers |
| **Vocabulary size** | Numero di token possibili |

## Termini Tecnici

| Termine | Significato |
|---------|-------------|
| **Quantization** | Comprimere pesi a precisione inferiore |
| **Dequantization** | Decomprimere per usarli |
| **MatVec** | Moltiplicazione matrice-vettore |
| **Logits** | Output grezzi prima del softmax |
| **Softmax** | Converte logits in probabilità |
| **Sampling** | Scelta del token finale |
| **Temperature** | Parametro per randomizzazione |
| **Top-k/P** | Tecniche di filtering per sampling |

## Termini Sistema

| Termine | Significato |
|---------|-------------|
| **KV Cache** | Cache di keys e values per attention |
| **Prefill** | Prima passata sul prompt |
| **Decode** | Generazione token uno alla volta |
| **Context** | Tutti i token processati finora |
| **Sliding window** | Finestra mobile per attention |
| **Context extension** | Estensione context oltre sliding window |

---

# Parte 7: Risorse per Approfondimento

## Documenti
- [README.md](README.md) - Documento principale
- [LICENSE](LICENSE) - Licenza e acknowledgements
- [tests/test-vectors/README.md](tests/test-vectors/README.md) - Test vectors

## Codice
- `ds4.c` - Motore principale
- `ds4.h` - Header e defines
- `ds4_cli.c` - CLI entry point
- `metal/*.metal` - Kernel Metal

## Link Esterni
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Progetto fondatore
- [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Formato file
- [HuggingFace model](https://huggingface.co/antirez/deepseek-v4-gguf) - Download modello
- [DeepSeek docs](https://deepseek.com) - Documentazione modello originale

---

# Conclusione

Questo documento è un punto di partenza. Il modo migliore per imparare è:
1. Leggere il codice
2. Eseguire il programma
3. Modificare qualcosa
4. Cercare di capire perché non funziona
5. Ripetere

Buon viaggio nell'inferenza locale!

---

## Nota sul Fork AMD HIP

Questo documento è stato creato come materiale di studio per comprendere il progetto ds4.c e le modifiche apportate nel fork [darchidev/ds4-hip](https://github.com/darchidev/ds4-hip).

Il fork aggiunge supporto sperimentale per GPU AMD tramite HIP/ROCm, consentendo l'esecuzione di ds4 su sistemi Linux con schede grafiche AMD (es. Ryzen AI Max+ 395 con Radeon 8060S).

Le modifiche al codice (backend HIP, integrazione nel flusso principale, decompressione tensori) sono state scritte con assistenza di programmazione automatica (AI).

---

*Questo documento è in evoluzione. Verrà aggiornato man mano che si approfondirà lo studio sul progetto.*