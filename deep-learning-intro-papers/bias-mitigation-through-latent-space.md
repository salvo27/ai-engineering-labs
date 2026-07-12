# Individuazione e mitigazione dei bias algoritmici attraverso la struttura latente appresa

## ABSTRACT
Alcune ricerche hanno evidenziato la vulnerabilità ai bias (pregiudizi sistematici che portano a risultati ingiusti o distorti) dei sistemi moderni basati sull'apprendimento automatico, soprattutto nei confronti di segmenti della società sottorappresentati nei dati di addestramento.

In questo lavoro, è stato sviluppato un algoritmo configurabile, per mitigare i bias nascosti e potenzialmente sconosciuti, all'interno dei dati di addestramento.

Questo algoritmo fonde il compito di apprendimento originale con i Variational Autoencoders (VAE, modelli che apprendono a comprimere i dati in uno spazio latente compatto e a ricostruirli, imparando così quali caratteristiche nascoste li descrivono), in modo tale da apprendere la struttura dello spazio latente (l'insieme delle caratteristiche non osservabili direttamente, ma che il modello scopre da solo, che spiegano le differenze tra i dati) all'interno del dataset, e utilizzando le strutture latenti apprese per riponderare l'importanza di alcuni punti durante l'addestramento (cioè dare più peso, durante l'apprendimento, agli esempi meno rappresentati).

Sebbene questo metodo sia generalizzabile a diverse modalità di dati e compiti di apprendimento, in questo lavoro l'algoritmo è stato utilizzato per affrontare il problema dei pregiudizi razziali e di genere nei sistemi di rilevamento facciale.

L'algoritmo viene valutato sul Pilot Parliaments Benchmark (un dataset di riferimento contenente immagini di volti di parlamentari di diversi paesi, creato apposta per testare l'equità dei sistemi di visione artificiale rispetto a genere ed etnia), e dimostriamo come con il nostro approccio di debiasing (correzione dei bias) aumentino le prestazioni complessive e diminuiscano i bias di categoria.

## 1 INTRODUZIONE
I sistemi di Machine Learning continuano a prendere sempre di più decisioni che impattano la vita quotidiana degli essere umani. Ad esempio ML ed AI vengono utilizzati per controllare veicoli autonomi end-to-end, determinare la durata delle pene per i condannati al carcere, stabilire l'ordine in cui una persona visualizza le notizie online, o persino diagnosticare e trattare malattie.

Lo sviluppo di sistemi AI equi e imparziali è fondamentale per prevenire effetti collaterali indesiderati e per garantire l'accettazione a lungo termine di questi algoritmi.

Anche il compito, che può apparentemente sembrare semplice, del riconoscimento facciale, si è dimostrato soggetto a grandi quantità di bias algoritmico in determinati gruppi demografici.

Ad esempio, Klare et al. hanno analizzato il sistema di rilevamento facciale delle forze dell'ordine statunitensi e si è rilevato significativamente più impreciso per le donne di carnagione scura tra i 18 e i 30 anni.

Sebbene i sistemi basati sul deep learning abbiano dimostrato prestazioni all'avanguardia in molti di questi compiti, è stato anche dimostrato che gli algoritmi addestrati con dati distorti portano a discriminazioni algoritmiche.

Recentemente sono emersi diversi benchmark con l'obiettivo di valutare l'equità di questi algoritmi e dei datasets. Tuttavia il problema dei datasets sbilanciati (in cui alcune categorie o gruppi sono rappresentati molto meno di altri) rimane, e il problema di integrare capacità di debiasing negli algoritmi è ancora irrisolto.

In questo paper, viene affrontata la sfida di integrare capacità di debiasing direttamente nel processo di addestramento del modello, in modo che si adatti autonomamente e senza supervisione (cioè senza che sia necessario etichettare manualmente quali esempi sono sottorappresentati) alle carenze nei dati d'addestramento.

Questo approccio prevede un algoritmo di deep learning end-to-end che apprende contemporaneamente il compito desiderato e la struttura latente sottostante ai dati d'addestramento.

Apprendere questa struttura in modo non supervisionato (senza etichette fornite dall'uomo, lasciando che il modello scopra da solo i pattern nei dati) ci permette di scoprire i bias nascosti o impliciti nei dati.

L'algoritmo, costruito sui VAE, è in grado di identificare esempi sotto rappresentati nei dati di addestramento e, di conseguenza, di aumentare la probabilità con cui l'algoritmo campiona questi dati.

L'algoritmo è applicabile a un'ampia gamma di compiti di computer vision ed è stato usato con successo anche per effettuare il debiasing di controllori di veicoli autonomi end-to-end.

In questo paper viene dimostrato come può essere utilizzato per correggere i bias di un sistema di rilevamento facciale addestrato su un dataset distorto, fornendo interpretazioni delle variabili latenti apprese (cioè rendendo comprensibile all'uomo cosa rappresentano concretamente le caratteristiche nascoste scoperte dal modello, come la posa o il tono della pelle), su cui l'algoritmo agisce attivamente.

Infine, verranno confrontate le prestazioni di questo algoritmo debiasato con un classificatore di deep learning standard (cioè non modificato per correggere i bias), valutando bias razziali e di genere sul dataset Pilot Parliaments Benchmark.

I contributi principali di questo paper sono:
- un nuovo algoritmo di debiasing regolabile che utilizza variabili latenti apprese per aggiustare le probabilità di campionamento dei singoli punti dati durante l'addestramento;
- un modello semi-supervisionato (che combina dati etichettati e non etichettati durante l'apprendimento) per apprendere simultaneamente un classificatore debiasato e le variabili latenti sottostanti che governano le classi di dati date;
- un'analisi del metodo, applicato al rilevamento facciale con dati di addestramento distorti, e una valutazione per misurare l'equità algoritmica tra diverse razze e generi.

## 2 LAVORI CORRELATI
Gli interventi che cercano di introdurre equità nella pipelines di machine learning rientrano principalmente in 3 categorie:
- quelle che utilizzano il pre-processing dei dati prima dell'addestramento;
- quelle in-processing durante l'addestramento;
- quelle post-processing, dopo l'addestramento.

Molti metodi di pre e in-processing si basano su dati generati artificialmente per correggere gli squilibri o su tecniche di resampling (ricampionamento, cioè la selezione ripetuta di alcuni esempi dal dataset per modificarne la composizione). Questi approcci, però, si concentrano quasi sempre sugli squilibri tra classi diverse (ad esempio: poche foto di gatti rispetto a molte di cani), non sulla variabilità interna a una stessa classe (ad esempio: nella classe "persone", alcune etnie, età o pose sono sottorappresentate rispetto ad altre), e non sfruttano alcuna informazione sulla struttura latente dei dati.

Il presente lavoro propone un nuovo approccio basato su VAE che effettua il resampling a partire dalla struttura latente dei dati, correggendo i bias automaticamente durante l'addestramento, senza bisogno di pre-processing o annotazioni manuali.

**Resampling per gli squilibri di classe**: Le tecniche di resampling si sono concentrate soprattutto sugli squilibri tra classi, non sui bias interni a una singola classe. Ad esempio, la duplicazione di esempi della classe minoritaria (cioè creare più copie degli esempi meno numerosi, per bilanciare il dataset) è stata usata come fase preliminare per correggere gli squilibri, ma non può adattarsi dinamicamente durante l'addestramento. Inoltre, estendere questi metodi ai bias interni a una classe richiederebbe di conoscere in anticipo la struttura latente dei dati, con conseguente necessità di annotazione manuale (cioè etichettare a mano le caratteristiche rilevanti, un processo costoso e soggettivo). Il seguente metodo, al contrario, corregge automaticamente la variabilità interna alle classi durante l'addestramento, apprendendo la struttura latente da zero e in modo non supervisionato.

**Generazione di dati corretti (debiased)**: alcuni lavori recenti utilizzano modelli generativi (modelli che imparano a creare nuovi dati simili a quelli osservati) o trasformazioni dei dati per produrre dataset di addestramento più equi rispetto a quello originale. Sattigeri et al. ad esempio, hanno usato una rete generativa avversaria (GAN, un modello composto da due reti neurali che competono tra loro: una genera dati sintetici, l'altra cerca di distinguerli da quelli reali) per ricostruire un dataset simile a quello di partenza ma più equo rispetto a certi attributi. Sono state proposte anche trasformazioni pre-processing per ridurre la discriminazione, ma si tratta di metodi statici, non adattivi, che non generano esempi realistici. Il seguente approccio, invece, non genera dati artificiali, ma seleziona un sottoinsieme del dataset originale, ricampionato per essere più rappresentativo.

**Clustering per individuare i bias**: anche il clustering k-means (una tecnica che raggruppa automaticamente i dati in un numero prefissato di gruppi simili tra loro, chiamati cluster) è stato utilizzato per individuare gruppi nei dati prima dell'addestramento, così da guidare il resampling verso un insieme più piccolo e rappresentativo di esempi. Questo approccio, tuttavia, non è scalabile a dati ad alta dimensionalità (cioè con moltissime caratteristiche, come i singoli pixel di un'immagine), non funziona quando non esiste una nozione chiara di "cluster", e richiede un pre-processing consistente per estrarre le caratteristiche rilevanti. Il seguente algoritmo supera questi limiti apprendendo direttamente la struttura latente tramite un approccio variazionale (basato sui VAE, che approssima matematicamente la distribuzione delle caratteristiche latenti nei dati).

## METODOLOGIA

### Impostazione del problema
