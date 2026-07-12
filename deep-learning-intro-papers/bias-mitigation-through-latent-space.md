# Individuazione e mitigazione dei bias algoritmici attraverso la struttura latente appresa

## ABSTRACT

Ricerche recenti hanno evidenziato la vulnerabilità ai bias (pregiudizi sistematici che portano a risultati ingiusti o distorti) dei sistemi moderni basati sull'apprendimento automatico, soprattutto nei confronti di segmenti della società sottorappresentati nei dati di addestramento.

In questo lavoro è stato sviluppato un nuovo algoritmo regolabile per mitigare i bias nascosti, e potenzialmente sconosciuti, presenti all'interno dei dati di addestramento.

Questo algoritmo fonde il compito di apprendimento originale con un Variational Autoencoder (VAE, un modello che apprende a comprimere i dati in uno spazio latente compatto e a ricostruirli, imparando così quali caratteristiche nascoste li descrivono), in modo da apprendere la struttura latente del dataset (l'insieme delle caratteristiche non osservabili direttamente, che il modello scopre da solo e che spiegano le differenze tra i dati) e da usare poi le distribuzioni latenti apprese per riponderare l'importanza di certi punti dati durante l'addestramento (cioè dare più peso, durante l'apprendimento, agli esempi meno rappresentati).

Sebbene questo metodo sia generalizzabile a diverse modalità di dati e compiti di apprendimento, in questo lavoro l'algoritmo viene utilizzato per affrontare il problema dei bias razziali e di genere nei sistemi di rilevamento facciale.

L'algoritmo viene valutato sul Pilot Parliaments Benchmark (PPB, un dataset di riferimento contenente immagini di volti di parlamentari di diversi paesi, creato apposta per testare l'equità dei sistemi di visione artificiale rispetto a genere e tono della pelle), e viene dimostrato come l'approccio di debiasing (correzione dei bias) proposto aumenti le prestazioni complessive e riduca il bias tra le categorie.

## 1 INTRODUZIONE

I sistemi di Machine Learning (ML) prendono decisioni che impattano sempre di più la vita quotidiana degli esseri umani e della società in generale. Ad esempio, ML e intelligenza artificiale (AI) vengono già utilizzati per controllare veicoli autonomi end-to-end (cioè con un'unica rete neurale che va direttamente dall'input, come le immagini della telecamera, all'output, come i comandi di sterzo, senza moduli intermedi progettati a mano), per determinare la durata delle pene per i condannati, per stabilire l'ordine in cui una persona visualizza le notizie online, o persino per diagnosticare e trattare pazienti.

Lo sviluppo e l'impiego di sistemi AI equi e imparziali è fondamentale per prevenire effetti collaterali indesiderati e per garantire l'accettazione a lungo termine di questi algoritmi.

Anche un compito apparentemente semplice come il riconoscimento facciale si è dimostrato soggetto a quantità estreme di bias algoritmico verso determinati gruppi demografici. Ad esempio, Klare et al. hanno analizzato il sistema di rilevamento facciale usato dalle forze dell'ordine statunitensi e hanno scoperto un'accuratezza significativamente più bassa per le donne di carnagione scura tra i 18 e i 30 anni. Questo è particolarmente preoccupante perché questi sistemi di riconoscimento facciale raramente operano in isolamento: spesso fanno parte di pipeline più ampie di sorveglianza o di individuazione di sospetti, dove un errore del sistema può avere conseguenze concrete sulle persone.

Sebbene i sistemi basati sul deep learning abbiano raggiunto prestazioni all'avanguardia in molti di questi compiti, è stato anche dimostrato che gli algoritmi addestrati con dati distorti producono discriminazione algoritmica. Recentemente sono emersi benchmark che quantificano la discriminazione e persino dataset progettati per valutare l'equità di questi algoritmi. Tuttavia, il problema dei dataset di addestramento fortemente sbilanciati (in cui alcune categorie o gruppi sono rappresentati molto meno di altri) e la questione di come integrare capacità di debiasing negli algoritmi di AI rimangono in gran parte irrisolti.

In questo paper viene affrontata la sfida di integrare le capacità di debiasing direttamente nel processo di addestramento del modello, in modo che si adatti automaticamente e senza supervisione (cioè senza che sia necessario etichettare manualmente quali esempi sono sottorappresentati) alle carenze dei dati di addestramento.

L'approccio si basa su un algoritmo di deep learning end-to-end che apprende contemporaneamente il compito desiderato (ad esempio il rilevamento facciale) e la struttura latente sottostante ai dati di addestramento. Apprendere questa distribuzione latente in modo non supervisionato (senza etichette fornite dall'uomo, lasciando che il modello scopra da solo i pattern nei dati) permette di far emergere i bias nascosti o impliciti presenti nei dati.

L'algoritmo, costruito sopra un VAE, è in grado di identificare gli esempi sottorappresentati nel dataset di addestramento e, di conseguenza, di aumentare la probabilità con cui l'algoritmo di apprendimento campiona questi punti dati.

L'algoritmo è applicabile a un'ampia gamma di compiti di computer vision ed è già stato usato con successo per il debiasing di controllori end-to-end di veicoli autonomi. In questo paper viene dimostrato come possa essere utilizzato anche per correggere i bias di un sistema di rilevamento facciale addestrato su un dataset distorto, fornendo inoltre interpretazioni delle variabili latenti apprese (cioè rendendo comprensibile all'uomo cosa rappresentano concretamente le caratteristiche nascoste scoperte dal modello, come la posa o il tono della pelle), che sono proprio le variabili su cui l'algoritmo agisce per correggere i bias.

Infine, le prestazioni del modello debiasato vengono confrontate con quelle di un classificatore di deep learning standard (cioè non modificato per correggere i bias), valutando i bias razziali e di genere sul dataset Pilot Parliaments Benchmark.

I contributi principali di questo paper sono:

- un nuovo algoritmo di debiasing regolabile che utilizza le variabili latenti apprese per aggiustare le probabilità di campionamento dei singoli punti dati durante l'addestramento;
- un modello semi-supervisionato (che combina apprendimento supervisionato su dati etichettati e apprendimento non supervisionato sulla struttura dei dati) per apprendere simultaneamente un classificatore debiasato e le variabili latenti sottostanti che governano le classi date;
- un'analisi del metodo applicato al rilevamento facciale con dati di addestramento distorti, e una valutazione per misurare l'equità algoritmica tra diverse etnie e generi.

## 2 LAVORI CORRELATI

Gli interventi che cercano di introdurre equità nelle pipeline di machine learning rientrano generalmente in una di queste 3 categorie:

- interventi di pre-processing, applicati ai dati prima dell'addestramento;
- interventi di in-processing, applicati durante l'addestramento;
- interventi di post-processing, applicati dopo l'addestramento.

Molti metodi di pre e in-processing si basano su dati generati artificialmente per correggere gli squilibri, oppure su tecniche di resampling (ricampionamento, cioè la selezione ripetuta di alcuni esempi dal dataset per modificarne la composizione). Questi approcci, però, si concentrano quasi sempre sugli squilibri tra classi diverse (ad esempio: poche foto di gatti rispetto a molte di cani), non sulla variabilità interna a una stessa classe (ad esempio: nella classe "volti", alcune etnie, età o pose sono sottorappresentate rispetto ad altre), e non sfruttano alcuna informazione sulla struttura delle caratteristiche latenti sottostanti.

L'apprendimento della struttura latente dei dati ha una lunga storia nel machine learning, che include tecniche come Expectation-Maximization (un algoritmo iterativo che stima i parametri di un modello in presenza di variabili nascoste), il topic modeling (che scopre i "temi" nascosti in collezioni di documenti), le latent-SVM e, più di recente, proprio i Variational Autoencoder.

Il presente lavoro propone un nuovo approccio basato su VAE che effettua il resampling a partire dalla struttura latente dei dati, corregge i bias automaticamente durante l'addestramento, e non richiede alcun pre-processing o annotazione manuale prima dell'addestramento o del test.

**Resampling per gli squilibri di classe**: le tecniche di resampling si sono concentrate soprattutto sugli squilibri tra classi, non sui bias interni a una singola classe. Ad esempio, la duplicazione di esempi della classe minoritaria (cioè creare più copie degli esempi meno numerosi, per bilanciare il dataset) è stata usata come fase di pre-processing per correggere gli squilibri, ma non può adattarsi dinamicamente durante l'addestramento. Inoltre, estendere questi metodi ai bias interni a una classe richiederebbe di conoscere a priori la struttura latente dei dati, con conseguente necessità di annotazione manuale delle caratteristiche desiderate (cioè etichettarle a mano, un processo costoso e soggettivo). L'approccio qui proposto, al contrario, corregge automaticamente la variabilità interna alle classi durante l'addestramento, apprendendo la struttura latente da zero e in modo non supervisionato.

**Generazione di dati corretti (debiased)**: alcuni lavori recenti utilizzano modelli generativi (modelli che imparano a creare nuovi dati simili a quelli osservati) o trasformazioni dei dati per produrre dataset di addestramento più equi rispetto a quello originale. Sattigeri et al., ad esempio, hanno usato una rete generativa avversaria (GAN, un modello composto da due reti neurali che competono tra loro: una genera dati sintetici, l'altra cerca di distinguerli da quelli reali) per ricostruire un dataset simile a quello di partenza ma più equo rispetto a certi attributi. Sono state proposte anche trasformazioni di pre-processing per ridurre la discriminazione, ma si tratta di metodi statici, non appresi in modo adattivo durante l'addestramento, e che non forniscono esempi di addestramento realistici. L'approccio qui proposto, invece, non si basa su dati generati artificialmente, ma utilizza un sottoinsieme del dataset originale, ricampionato per essere più rappresentativo.

**Clustering per individuare i bias**: anche il clustering k-means (una tecnica che raggruppa automaticamente i dati in un numero prefissato di gruppi simili tra loro, chiamati cluster) è stato utilizzato per individuare gruppi nei dati prima dell'addestramento, così da guidare il resampling verso un insieme più piccolo e rappresentativo di esempi. Questo approccio, tuttavia, non è scalabile a dati ad alta dimensionalità (cioè con moltissime caratteristiche, come i singoli pixel di un'immagine), non funziona quando non esiste una nozione chiara di "cluster" nei dati, e richiede un pre-processing consistente per estrarre le caratteristiche rilevanti. L'algoritmo qui proposto supera questi limiti apprendendo direttamente la struttura latente sottostante tramite un approccio variazionale (basato sui VAE, che approssima matematicamente la distribuzione delle caratteristiche latenti nei dati).

## 3 METODOLOGIA

### 3.1 Impostazione del problema

Consideriamo il problema della classificazione binaria (in cui il modello deve assegnare a ogni esempio una tra due possibili etichette, ad esempio "volto" o "non volto"). Abbiamo un insieme di coppie di dati di addestramento $\mathcal{D}_{train} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{n}$, formato da feature $x \in \mathbb{R}^m$ (nel nostro caso, i pixel di un'immagine) ed etichette $y \in \mathbb{R}^d$.

L'obiettivo è trovare una funzione $f: X \rightarrow Y$, parametrizzata da $\theta$ (i pesi della rete neurale), che minimizzi una certa loss $\mathcal{L}(\theta)$ (la funzione di costo, che misura quanto il modello sbaglia) sull'intero dataset di addestramento. In altre parole, cerchiamo di risolvere il seguente problema di ottimizzazione:

$$\theta^* = \arg\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}_i(\theta) \tag{1}$$

Dato un nuovo esempio di test $(x, y)$, il classificatore dovrebbe idealmente produrre un output $\hat{y} = f_\theta(x)$ "vicino" a $y$, dove la nozione di vicinanza è definita dalla loss originale.

Assumiamo ora che ogni punto dati abbia anche un vettore latente continuo associato $z \in \mathbb{R}^k$, che cattura le caratteristiche nascoste e sensibili dell'esempio (nel caso dei volti: tono della pelle, genere, età, posa e così via). Possiamo allora formalizzare la nozione di classificatore distorto come segue:

**Definizione 1.** *Un classificatore $f_\theta(x)$ è distorto (biased) se la sua decisione cambia quando viene esposto a input aggiuntivi di caratteristiche sensibili. In altre parole, un classificatore è equo rispetto a un insieme di caratteristiche latenti $z$ se: $f_\theta(x) = f_\theta(x, z)$.*

Ad esempio, quando si decide se un'immagine contiene un volto oppure no, il colore della pelle, il genere o persino l'età della persona sono tutte variabili latenti sottostanti, e cambiare uno qualsiasi dei loro valori non dovrebbe alterare la decisione finale del classificatore.

Per garantire l'equità di un classificatore rispetto a queste variabili latenti, il dataset dovrebbe contenere campioni distribuiti in modo approssimativamente uniforme sullo spazio latente. In altre parole, la distribuzione di addestramento stessa non dovrebbe sovrarappresentare una certa categoria a scapito di altre.

Attenzione: questo è diverso dal dire che il dataset debba essere bilanciato rispetto alle classi (cioè contenere circa lo stesso numero di volti e di non-volti). Quello che si sta dicendo è che, all'interno di una singola classe, anche le variabili latenti non osservate dovrebbero essere bilanciate. In questo modo tutte le istanze di una stessa classe verranno trattate equamente dal classificatore: anche portando una variabile latente all'estremo opposto (ad esempio il tono della pelle da chiaro a scuro), l'accuratezza del classificatore non dovrebbe cambiare.

Inoltre, se disponiamo di un test set etichettato rispetto allo spazio delle variabili latenti sensibili $z$, possiamo misurare il bias del classificatore calcolando la sua accuratezza su ciascuna delle categorie sensibili (ad esempio i diversi toni della pelle). Mentre l'accuratezza complessiva del classificatore è la media delle accuratezze su tutte le categorie sensibili, il bias è la varianza delle accuratezze tra le diverse realizzazioni di queste categorie (ad esempio volti chiari contro volti scuri).

Questa definizione è molto intuitiva: se un classificatore funziona ugualmente bene indipendentemente dal valore di una specifica variabile latente (ad esempio il tono della pelle), la varianza delle sue accuratezze sarà zero, e il classificatore si dirà non distorto rispetto a quella variabile. Se invece alcuni valori della variabile latente fanno funzionare il classificatore meglio o peggio, la varianza delle accuratezze aumenterà, e con essa il bias complessivo del classificatore.

Sarebbe possibile usare un insieme di variabili sensibili definite manualmente per garantire una rappresentazione equa durante l'addestramento, ma questo richiederebbe un'annotazione manuale di ogni variabile sull'intero dataset, molto costosa in termini di tempo. In più, questo approccio è soggetto al potenziale bias umano nella scelta di quali variabili considerare sensibili e quali no. In questo lavoro il problema viene risolto apprendendo le variabili latenti della classe in modo completamente non supervisionato, per poi usarle per ricampionare adattivamente il dataset durante l'addestramento. Nella prossima sottosezione viene descritta l'architettura usata per apprendere le variabili latenti.

### 3.2 Apprendere la struttura latente con i Variational Autoencoder

In questo lavoro le variabili latenti di una classe vengono *apprese* in modo completamente non supervisionato, e poi usate per ricampionare adattivamente il dataset durante l'addestramento. Per farlo, viene proposta un'estensione dell'architettura VAE: il debiasing-VAE (DB-VAE).

La parte encoder del VAE (la rete che comprime l'input nello spazio latente) apprende un'approssimazione $q_\phi(z|x)$ della vera distribuzione delle variabili latenti dato un punto dati. A differenza delle architetture VAE classiche, vengono introdotte anche $d$ variabili di output aggiuntive, con $\hat{y} \in \mathbb{R}^d$. Con $k$ variabili latenti e $d$ variabili di output, l'encoder produce $2k + d$ attivazioni: $2k$ corrispondono a $\mu \in \mathbb{R}^k$ e $\Sigma = Diag[\sigma^2] \succ 0$ (la media e la varianza che definiscono la distribuzione di $z$; nei VAE, infatti, l'encoder non produce un singolo punto nello spazio latente, ma una distribuzione gaussiana da cui campionare), più le $d$ attivazioni dell'output $\hat{y}$.

Il punto chiave è questo: per continuare ad apprendere anche il compito supervisionato originale, le $d$ variabili di output vengono supervisionate esplicitamente (cioè confrontate con le etichette vere durante l'addestramento). Questo trasforma il VAE da modello completamente non supervisionato a modello semi-supervisionato: alcune variabili latenti vengono apprese implicitamente cercando di ricostruire l'input, mentre le altre vengono supervisionate esplicitamente per un compito specifico (ad esempio la classificazione). Nel caso di un classificatore binario ($\hat{y} \in \{0, 1\}$), il modello DB-VAE apprende una codifica di $k$ variabili latenti, cioè $\{z_i\}_{i \in \{1,k\}}$, più una singola variabile dedicata alla classificazione: $z_0 = \hat{y}$.

Una rete decoder speculare all'encoder viene poi usata per ricostruire l'input a partire dallo spazio latente, approssimando $p_\theta(x|z)$. I VAE usano il trucco della riparametrizzazione (reparameterization trick) per poter calcolare i gradienti attraverso il passaggio di campionamento, che di per sé non sarebbe derivabile: si campiona $\epsilon \sim \mathcal{N}(0, I)$ da una gaussiana standard e si calcola $z = \mu(x) + \Sigma^{\frac{1}{2}}(x) \circ \epsilon$. In questo modo la casualità viene isolata in $\epsilon$ e i gradienti possono fluire attraverso $\mu$ e $\Sigma$. La ricostruzione decodificata rende possibile l'apprendimento non supervisionato delle variabili latenti durante l'addestramento, ed è quindi necessaria per il debiasing automatico dei dati.

La rete viene addestrata end-to-end tramite backpropagation con una loss a tre componenti: una loss supervisionata per la classificazione, una loss di ricostruzione e una loss latente per le variabili non supervisionate. Per un compito di classificazione binaria, ad esempio:

- la loss supervisionata $\mathcal{L}_y(y, \hat{y})$ è la cross-entropy (la loss standard per la classificazione, che penalizza il modello quando assegna bassa probabilità alla classe corretta);
- la loss di ricostruzione $\mathcal{L}_x(x, \hat{x})$ è la norma $L_p$ tra l'input e l'output ricostruito (cioè una misura di distanza pixel per pixel tra immagine originale e ricostruita);
- la loss latente $\mathcal{L}_{KL}(\mu, \sigma)$ è la divergenza di Kullback-Leibler (una misura di quanto la distribuzione latente appresa si discosta dalla gaussiana standard usata come riferimento; serve a mantenere lo spazio latente regolare e ben organizzato).

La loss totale è una combinazione pesata di queste tre componenti:

$$\mathcal{L}_{TOTAL} = c_1 \underbrace{\left[\sum_{i \in \{0,1\}} y_i \log\left(\frac{1}{\hat{y}_i}\right)\right]}_{\mathcal{L}_y(y,\hat{y})} + c_2 \underbrace{\left[\|x - \hat{x}\|_p\right]}_{\mathcal{L}_x(x,\hat{x})} + c_3 \underbrace{\left[\frac{1}{2}\sum_{j=0}^{k-1}(\sigma_j + \mu_j^2 - 1 - \log(\sigma_j))\right]}_{\mathcal{L}_{KL}(\mu,\sigma)} \tag{2}$$

dove $c_1, c_2, c_3$ sono i coefficienti di peso che regolano l'importanza relativa di ciascuna delle tre loss.

Come termine di confronto, il modello baseline (il modello di riferimento standard con cui confrontare i risultati) usato per il compito ha un'architettura simile al DB-VAE, ma senza le variabili latenti non supervisionate e senza la rete decoder, e viene addestrato usando solo la loss supervisionata.

Una precisazione importante riguarda gli esempi appartenenti a classi che *non* si vogliono debiasare. Nel problema del rilevamento facciale, ad esempio, interessa soprattutto che il dataset positivo (i volti) sia equo e non distorto, mentre importa poco correggere i bias degli esempi negativi in cui non c'è alcun volto. Per questi campioni negativi, i gradienti provenienti dal decoder e dallo spazio latente vengono bloccati e non retropropagati. In pratica, per queste classi si addestra solo l'encoder a ottimizzare la loss supervisionata.

### 3.3 Algoritmo per il debiasing automatico

In questa sezione viene presentato l'algoritmo di resampling adattivo dei dati di addestramento basato sulla struttura latente appresa dal modello DB-VAE. Scartando le regioni sovrarappresentate dello spazio latente in proporzione alla loro frequenza, si aumenta la probabilità di selezionare per l'addestramento i dati più rari. Questo avviene in modo adattivo, mentre le variabili latenti stesse vengono apprese durante l'addestramento. L'approccio di debiasing tiene quindi conto della distribuzione completa delle caratteristiche sottostanti nei dati di addestramento.

Il dataset di addestramento viene fatto passare attraverso la rete encoder, che fornisce una stima $Q(z|X)$ della distribuzione latente. L'obiettivo è aumentare la frequenza relativa dei punti dati rari, campionando di più le regioni sottorappresentate dello spazio latente. Per farlo, la distribuzione dello spazio latente viene approssimata con un istogramma $\hat{Q}(z|X)$, la cui dimensionalità è definita dal numero di variabili latenti $k$.

Qui però nasce un problema pratico: un istogramma congiunto su $k$ dimensioni diventa rapidamente intrattabile al crescere di $k$ (servirebbe un numero di celle esponenziale in $k$, la cosiddetta maledizione della dimensionalità). Per aggirare il problema si semplifica ulteriormente usando istogrammi indipendenti per approssimare la distribuzione congiunta. In pratica, si definisce un istogramma indipendente $\hat{Q}_i(z_i|X)$ per ciascuna variabile latente $z_i$:

$$\hat{Q}(z|X) \propto \prod_i \hat{Q}_i(z_i|X) \tag{3}$$

Questo permette di approssimare $Q(z|X)$ in modo semplice, basandosi sulla distribuzione di frequenza di ciascuna variabile latente appresa.

Infine, viene introdotto un singolo parametro $\alpha$ per regolare il grado di debiasing applicato durante l'addestramento. La probabilità di selezionare un punto dati $x$ viene definita come $\mathcal{W}(z(x)|X)$, parametrizzata dal parametro di debiasing $\alpha$:

$$\mathcal{W}(z(x)|X) \propto \prod_i \frac{1}{\hat{Q}_i(z_i(x)|X) + \alpha} \tag{4}$$

L'intuizione dietro questa formula: se un esempio cade in una regione molto frequente dello spazio latente, $\hat{Q}_i$ è grande e quindi la sua probabilità di essere selezionato è bassa; se cade in una regione rara, $\hat{Q}_i$ è piccolo e la probabilità di selezione è alta. Il parametro $\alpha$ al denominatore smorza questo effetto.

Lo pseudocodice per l'addestramento del DB-VAE è riportato nell'Algoritmo 1. A ogni epoca (un passaggio completo su tutto il dataset di addestramento), tutti gli input $x$ del dataset originale $X$ vengono propagati attraverso il modello per valutare le corrispondenti variabili latenti $z(x)$, e gli istogrammi $\hat{Q}_i(z_i(x)|X)$ vengono aggiornati di conseguenza. Durante l'addestramento, ogni nuovo batch viene estratto mantenendo gli input $x$ del dataset originale $X$ con probabilità $\mathcal{W}(z(x)|X)$.

Addestrare sul batch debiasato costringe il classificatore verso una scelta di parametri che funziona meglio nei casi rari, senza un forte deterioramento delle prestazioni sugli esempi di addestramento comuni. Il punto più importante è che il debiasing non viene specificato manualmente in anticipo, ma si basa sulle variabili latenti *apprese*.

**Algoritmo 1** - Resampling adattivo per il debiasing automatico dell'architettura DB-VAE

**Richiede:** dati di addestramento $\{X, Y\}$, dimensione del batch $b$

1. Inizializza i pesi $\{\phi, \theta\}$
2. **per** ogni epoca $E_t$ **fai**
3. &nbsp;&nbsp;&nbsp;&nbsp;Campiona $z \sim q_\phi(z|X)$
4. &nbsp;&nbsp;&nbsp;&nbsp;Aggiorna $\hat{Q}_i(z_i(x)|X)$
5. &nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{W}(z(x)|X) \leftarrow \prod_i \frac{1}{\hat{Q}_i(z_i(x)|X) + \alpha}$
6. &nbsp;&nbsp;&nbsp;&nbsp;**finché** $iter < \frac{n}{b}$ **fai**
7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Campiona $x_{batch} \sim \mathcal{W}(z(x)|X)$
8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$L(\phi, \theta) \leftarrow \frac{1}{b}\sum_{i \in x_{batch}} \mathcal{L}_i(\phi, \theta)$
9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Aggiorna: $[w \leftarrow w - \eta\nabla_{\phi,\theta}\mathcal{L}(\phi, \theta)]_{w \in \{\phi,\theta\}}$
10. &nbsp;&nbsp;&nbsp;&nbsp;**fine finché**
11. **fine per**

Intuitivamente, il parametro $\alpha$ regola il grado di debiasing. Quando $\alpha \rightarrow 0$, il training set sottocampionato tende a diventare uniforme rispetto alle variabili latenti $z$ (debiasing massimo: tutte le regioni dello spazio latente vengono campionate ugualmente). Quando $\alpha \rightarrow \infty$, il training set sottocampionato tende a un campione casuale uniforme del dataset di addestramento originale (cioè nessun debiasing: il termine $\alpha$ domina il denominatore e tutte le probabilità diventano uguali).

## 4 ESPERIMENTI

Per validare l'algoritmo di debiasing su un problema reale con un impatto sociale significativo, viene appreso un rilevatore facciale debiasato usando dati di addestramento potenzialmente distorti. In questa sezione viene definito il problema del rilevamento facciale, vengono descritti i dataset usati, e vengono delineati l'addestramento del modello, il debiasing e la valutazione.

Per il problema del rilevamento facciale, viene dato un insieme di coppie di dati di addestramento $\mathcal{D}_{train} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{n}$, dove $x^{(i)}$ sono i valori grezzi dei pixel di una porzione di immagine (image patch) e $y^{(i)} \in \{0, 1\}$ sono le rispettive etichette, che indicano la presenza di un volto.

L'obiettivo è garantire che l'insieme degli esempi positivi usati per addestrare il classificatore sia equo e non distorto. I dati di addestramento positivi potrebbero infatti essere distorti rispetto ad alcuni attributi, come il tono della pelle, nel senso che particolari valori di questi attributi potrebbero apparire più o meno frequentemente di altri.

Negli esperimenti viene quindi addestrato un modello DB-VAE completo per apprendere la struttura latente sottostante alle immagini positive (i volti), e viene usato l'approccio di resampling adattivo dell'Algoritmo 1 per debiasare il modello rispetto alle caratteristiche facciali. Per gli esempi negativi viene addestrata solo la parte encoder della rete, come descritto nella Sezione 3. Le prestazioni dei modelli debiasati vengono valutate rispetto ai classificatori standard (distorti) sul dataset PPB, fornendo stime della precisione e del bias di ciascun modello come metriche di prestazione.

### 4.1 Dataset

I classificatori vengono addestrati su un dataset di $n = 4 \times 10^5$ immagini, composto da $2 \times 10^5$ esempi positivi (immagini di volti) e altrettanti negativi (immagini di non-volti), suddivisi rispettivamente all'80% e al 20% in set di addestramento e di validazione. Gli esempi positivi provengono dal dataset CelebA (un grande dataset pubblico di volti di celebrità) e sono stati ritagliati in formato quadrato in base al riquadro del volto annotato. Gli esempi negativi provengono dal dataset ImageNet, da un'ampia varietà di categorie non umane. Tutte le immagini sono state ridimensionate a 64 × 64 pixel.

Dopo l'addestramento, l'algoritmo di debiasing viene valutato sul test set PPB, che consiste in immagini di 1270 parlamentari uomini e donne di vari paesi africani ed europei. Le immagini sono coerenti in termini di posa, illuminazione ed espressione facciale, e il dataset presenta parità sia nel tono della pelle sia nel genere. Il genere di ogni volto è annotato con le etichette "Maschio" e "Femmina". Le annotazioni del tono della pelle si basano sulla scala di Fitzpatrick (un sistema dermatologico di classificazione della pelle in base alla sua reazione al sole), con ogni immagine etichettata come "Più chiara" o "Più scura".

### 4.2 Addestramento dei modelli

Per il compito classico di rilevamento facciale viene addestrata una rete neurale convoluzionale con quattro layer convoluzionali in sequenza (filtri 5 × 5 con stride 2 × 2, dove lo stride è il passo con cui il filtro scorre sull'immagine) per l'estrazione delle feature. La classificazione finale avviene tramite due layer fully connected aggiuntivi (layer in cui ogni neurone è collegato a tutti quelli del layer precedente), rispettivamente con 1000 e 1 neuroni. Tutti i layer della rete usano l'attivazione ReLU e la batch normalization (una tecnica che normalizza le attivazioni per stabilizzare e velocizzare l'addestramento).

L'architettura DB-VAE condivide questa stessa rete di classificazione per l'encoder, tranne per il layer fully connected finale, che ora produce $k$ variabili latenti aggiuntive per un totale di $2k + 1$ attivazioni. Un decoder, speculare all'encoder con 2 layer fully connected e 4 layer deconvoluzionali (layer che effettuano l'operazione inversa della convoluzione, ricostruendo un'immagine a partire da una rappresentazione compressa), viene poi usato per ricostruire l'immagine di input originale. I modelli vengono addestrati minimizzando la loss empirica definita nell'Eq. 2, con loss di ricostruzione $L_2$.

Negli esperimenti, inoltre, tutti i gradienti provenienti dal decoder vengono bloccati quando $y = 0$, cioè per gli esempi negativi, poiché si vuole debiasare solo gli esempi positivi di volti. Oltre alla rete di classificazione standard senza debiasing, sono stati addestrati modelli DB-VAE con gradi crescenti di debiasing, definiti dal parametro $\alpha$, per 50 epoche, valutandone le prestazioni sul validation set. Ogni modello è stato riaddestrato da zero 5 volte per una maggiore robustezza statistica dei risultati.

### 4.3 Debiasing automatico dei sistemi di rilevamento facciale

![Debiasing](../assets/papers/bias/debiasing2.png)

In questa sezione viene esplorato l'output dell'algoritmo di debiasing e viene fornita una valutazione estesa dei modelli appresi sul dataset PPB. Consideriamo le probabilità di resampling $\mathcal{W}(z(x)|X)$ che emergono dall'apprendimento di un modello debiasato. Al crescere della probabilità di resampling, il numero di punti dati nel bin corrispondente (l'intervallo dell'istogramma) diminuisce, il che suggerisce che le immagini con maggiore probabilità di essere ricampionate sono quelle caratterizzate da feature "rare".

![Automatic Debiasing](../assets/papers/bias/automatic-debiasing.png)

In effetti, al crescere della probabilità di resampling, le immagini corrispondenti diventano più diversificate. Questa osservazione è ulteriormente confermata considerando i dieci volti del training set con le probabilità di resampling più basse e più alte (Fig. 4B e 4C rispettivamente). I dieci volti con la probabilità più bassa appaiono piuttosto uniformi: tono della pelle, colore dei capelli, sguardo frontale e colore dello sfondo sono coerenti tra loro. Al contrario, i dieci volti con la probabilità più alta mostrano caratteristiche più rare, come copricapi o occhiali, sguardo inclinato, ombre e pelle più scura. Nel complesso, questi risultati indicano che l'algoritmo identifica e poi ricampiona attivamente i punti dati con feature più rare e diversificate, sulla base di una rappresentazione latente appresa.

È stato osservato che il DB-VAE riesce ad apprendere caratteristiche facciali come il tono della pelle, la presenza di capelli e l'azimut (l'angolo di rotazione orizzontale del volto), così come altre caratteristiche come genere ed età. Lo si può verificare perturbando lentamente il valore di una singola variabile latente e facendo passare la codifica risultante attraverso il decoder (Fig. 5A): l'immagine ricostruita cambia gradualmente proprio lungo quella caratteristica. Questo supporta l'ipotesi che l'algoritmo DB-VAE sia capace di debiasare rispetto a tali caratteristiche, dato che le probabilità di resampling sono definite direttamente sulle distribuzioni di probabilità delle *singole* variabili latenti apprese (Alg. 1).

Per valutare le prestazioni dell'approccio di debiasing è stata utilizzata come metrica l'accuratezza di classificazione (valore predittivo positivo), testando i modelli sul dataset PPB. Per questa valutazione, da ogni immagine sono state estratte delle porzioni usando finestre scorrevoli (sliding windows) di dimensioni variabili, che sono state poi date in input ai modelli addestrati. Il sistema restituisce un riscontro positivo se il classificatore identifica un volto in almeno una delle sotto-porzioni dell'immagine.

Per dimostrare il debiasing rispetto a specifiche caratteristiche latenti, le prestazioni di classificazione sono state quantificate sulle singole categorie demografiche. In particolare, sono stati considerati il tono della pelle (chiaro/scuro) e il genere (maschio/femmina). Indichiamo con $\mathcal{A}$ l'insieme delle accuratezze di classificazione di un modello su ciascuna delle quattro classi intersezionali (le quattro combinazioni: maschio scuro, femmina scura, maschio chiaro, femmina chiara). Sono state confrontate le accuratezze dei modelli addestrati con e senza debiasing, sia sulle singole categorie demografiche sia sul dataset PPB nel complesso, mostrando anche l'effetto del parametro di debiasing $\alpha$ sulle prestazioni (Fig. 5).

Ricordiamo che l'assenza di debiasing corrisponde al limite $\alpha \rightarrow \infty$, dove si campiona uniformemente dal *training set originale* senza apprendere le variabili latenti. Al contrario, $\alpha \rightarrow 0$ corrisponde a campionare da una distribuzione uniforme sullo *spazio latente*. Le barre di errore (errore standard della media) sono fornite per visualizzare la significatività statistica delle differenze tra i modelli addestrati.

Come mostrato in Fig. 5, una maggiore forza di debiasing (cioè un $\alpha$ decrescente) ha aumentato significativamente l'accuratezza di classificazione sui soggetti "maschi scuri", in linea con l'ipotesi che il resampling adattivo delle istanze rare (ad esempio i volti scuri) nei dati di addestramento riduca la discriminazione algoritmica. Questo suggerisce che l'algoritmo può debiasare rispetto a una caratteristica qualitativa come il tono della pelle, con implicazioni sociali significative per il miglioramento dell'equità nei sistemi di rilevamento facciale.

Al contrario di quanto osservato per i volti maschili scuri, l'accuratezza di classificazione sui volti "maschi chiari" è rimasta pressoché costante sia per i modelli distorti sia per quelli debiasati. Inoltre, l'accuratezza sui soggetti maschi chiari è risultata più alta rispetto agli altri tre gruppi, in linea con la letteratura precedente. Questo suggerisce che l'algoritmo di debiasing non sacrifica le prestazioni sulle categorie che hanno già una precisione elevata. Un dettaglio importante: l'accuratezza alta e quasi costante suggerisce che un qualsiasi modello di classificazione addestrato sul dataset CelebA rischia di essere distorto a favore dei soggetti maschi chiari, il che rafforza ulteriormente la necessità di approcci che cerchino di ridurre questi bias.

**Tabella 1: accuratezza e bias sul test set PPB.**

| | $\mathbb{E}[\mathcal{A}]$ (Precisione) | $Var[\mathcal{A}]$ (Misura del bias) |
|---|---|---|
| Senza debiasing | 95.13 | 28.84 |
| $\alpha = 0.1$ | 95.84 | 25.43 |
| $\alpha = 0.05$ | 96.47 | 18.08 |
| $\alpha = 0.01$ | 97.13 | 9.49 |
| $\alpha = 0.001$ | **97.36** | **9.43** |

Sebbene il DB-VAE abbia migliorato significativamente l'accuratezza sui maschi scuri, questa non ha mai raggiunto quella dei maschi chiari. Nonostante il debiasing dei dati di addestramento rispetto a variabili latenti come il tono della pelle, nel dataset ci sono intrinsecamente meno esempi di volti maschili scuri. Il modello è semplicemente limitato dalla scarsità di questi esempi, ma va notato che aumentare la dimensione complessiva del dataset di addestramento potrebbe mitigare ulteriormente questo effetto.

Le tendenze principali delle prestazioni complessive del DB-VAE sono riassunte nella Tabella 1. Come confermato dalla Fig. 5, la precisione complessiva $\mathbb{E}[\mathcal{A}]$ è aumentata al crescere della forza di debiasing (cioè al diminuire di $\alpha$). Inoltre, è stata osservata una diminuzione della varianza delle accuratezze tra le categorie, indice di un bias ridotto con un debiasing più forte. Nel complesso, questi risultati suggeriscono che il DB-VAE effettua un debiasing efficace.

## 5 CONCLUSIONE

In questo paper viene proposto un nuovo algoritmo di debiasing regolabile che aggiusta le probabilità di campionamento dei singoli punti dati durante l'addestramento. Apprendendo le variabili latenti sottostanti in modo completamente non supervisionato, l'approccio può scalare a dataset di grandi dimensioni e debiasare rispetto a caratteristiche latenti senza mai doverle etichettare a mano nel training set.

L'approccio viene applicato al rilevamento facciale per promuovere l'equità algoritmica riducendo i bias nascosti nei dati di addestramento. Dato un dataset di addestramento distorto, i modelli debiasati mostrano una maggiore accuratezza di classificazione e un bias di categoria ridotto rispetto a etnia e genere, in confronto ai classificatori standard. Viene infine fornito un algoritmo concreto per il debiasing, insieme a un'implementazione open source del modello.

Lo sviluppo e l'impiego di sistemi AI equi e imparziali è fondamentale per prevenire discriminazioni indesiderate e per garantire l'accettazione a lungo termine di questi algoritmi. Gli autori prevedono che l'approccio proposto possa servire come strumento aggiuntivo per promuovere l'equità algoritmica sistematica dei moderni sistemi di AI.