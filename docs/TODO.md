# Edições a serem feitas antes da submissão ao CBA 2026

*Sobre este documento*:
Leitura da versão atual do artigo (5 de abril de 2026)
gerou a seguinte lista de tarefas, separadas por seção.

Projeto no Overleaf:
https://www.overleaf.com/project/69604c0a3ffb98736449e03e

## Título
- Remover menção a flatness
- Rodapé/"estrela" ao final: 
	- talvez não seja necessária, que o artigo não puder ser considerado parte do MIPson
	- Espaço horizontal parece exagerado

## Autores
- Remover Eugênio: não usamos o filtro adaptativo
- Remover a profa. Marcela se o artigo não for parte do MIPson

## Abstract
- Remover menção a flatness
- Deixar claro que uma das novidades do artigo é a consideração dos efeitos do tremor em outras juntas (ombro, cotovelo) na junta controlada (punho)
- Explicitar os números obtidos na amplitude suppression ratio
- Se cabível, mencionar a forma como o PID foi ajustado
- Remover menção ao motor, substituindo por FES caso essa seja usada

## Keywords
- Trocar "Flatness Control" por "Voluntary motion tracking" ou similar mais adequado

## 1 Introduction
Está vazia, mas já com as questões-guia colocadas no próprio artigo.
Pontos importantes:
- Ruído de medida não é um fator crítico, então pode-se remover as perguntas a respeito.
- Flatness não é mais usada, então pode-se remover a pergunta a respeito.
- Deve-se deixar claro na contribuição a modelagem de ordem maior 
- Citar pelo menos 6 formas de suprimir tremor patológico:
	1. medicação
	2. deep brain stimulation (DBS)
	3. peripherical electrical stimulation (PES)
	4. functional electrical stimulation (FES)
	5. órteses passivas
	6. órteses ativas (controle ativo)

Obs.: organização temática das referências é feita ao final deste documento.
	
## 2 Theorical Framework
### 2.1 Upper limb model
- Incluir o diagrama faltante

### 2.2 Adaptive Filtering
- Remover seção e trocar por seção sobre o método escolhido para estimar o movimento voluntário (possivelmente colocar essa seção após explicação sobre ADRC)

### 2.3 Error based ADRC
- Incluir equações do ADRC
- Incluir diagrama do ADRC
- Incluir explicação sobre o ajuste dos parâmetros do ADRC

## 3 Methodology
- Incluir seção "3.0" sobre modelagem do sinal (tremor + voluntário + ruído)
- Na nova seção, incluir diagrama com a adição do tremor e do ruído, e explicar como o tremor é modelado (soma de senos) e como o ruído é modelado (ruído branco gaussiano)

### 3.1 Closed loop setup
- Figura 3: remover bloco atuador, dicussão sobre atuador, e detalhar bloco de controle (tomar inspiração nos vídeos tutoriais de ADRC)

### 3.2 Control design
- Incluir as equações de controle (PID e ADRC), caso já não tenham sido incluídas na seção de teoria

### 3.3 Simulational setup

### 3.4 Parameters
- Incluir os intervalos de amostragem das variáveis que alteram a modelagem do sistema
- Identificar o eixo de rotação dos momentos de inércia

## 4 Results and Discussions
- Comentar o método utilizado para rodar a simulação: 
Runge-Kutta de quarta ordem, com passo de 0.001s, e duração de 6s

### 4.1 Scenario 1: joints at rest
- Configurar as condições iniciais
- Figura 5: Incluir gráfico com as respostas de todos os sistemas para cada controle
- Preencher tabela 4 com os resultados das simulações

### 4.2 Scenario 2: joints in motion
- Configurar as condições iniciais
- Figura 6: Incluir gráfico com os perfis de torque voluntário
- Figura 7: Incluir gráfico com as respostas de todos os sistemas para cada controle

## 5 Conclusion
- Incluir as conclusões do artigo
- Trabalhos futuros: comentar que métodos de predição de tremor estão cada vez mais populares

## References
- Remover a referência do filtro adaptativo
- Organização das referências:
	- Revisões bibliográficas sobre caracterização de tremor
		- Tremor (2009)
		- The differential diagnosis and treatment of tremor; [Differenzialdiagnose und therapie des tremors] (2014)
		- The clinical and electrophysiological investigation of tremor (2022)
	
	- Revisões bibliográficas sobre supressão de tremor (geral)
		- An overview and categorization of dynamic arm supports for people with decreased arm function
		- Design of Passive Dynamic Absorbers to Attenuate Pathological Tremor of Human Upper Limb (2020)
		- A Review on Wearable Technologies for Tremor Suppression (2021)
		- Tremor-Suppression Orthoses for the Upper Limb: Current Developments and Future Challenges (2021)
		- Medical Devices for Tremor Suppression: Current Status and Future Directions (2021)
		- A Review of Techniques Used to Suppress Tremor (2022)
		- Deep brain stimulation for the treatment of tremor (2022)
		- Improving functional disability in patients with tremor: A clinical perspective of the efficacies, considerations, and challenges of assistive technology (2022)
		- Peripherical Electrical Stimulation for Parkinsonian Tremor: A Systematic Review (2022)
		- Diagnosis and Treatment of Tremor in Parkinson’s Disease Using Mechanical Devices (2023)
		- Management of essential tremor deep brain stimulation-induced side effects (2024)
		- Upper limb intention tremor assessment: opportunities and challenges in wearable technology (2024)
		- The Application of Deep Brain Stimulation on Multiple Sclerosis Tremors and the Emerging Targets: A Mini-Review (2025)
		- Technological interventions for the suppression of hand tremors: A literature review (2025)
		- Gait analysis and treatment strategies for motor symptoms in Parkinson's disease: a comprehensive review (2025)
	
	- Métodos para supressão de tremor usando controle passivo
		- Tremor Reduction at the Palm of a Parkinson’s Patient Using Dynamic Vibration Absorber (2016)
		- Involuntary Tremor Controlled Using Mechanical Means (2017)
		- Parametric study of an enhanced passive absorber used for tremor suppression (2018)
		- METAESTRUTURA INTELIGENTE PROGRAMÁVEL PARA CONTROLE PASSIVO DE TREMORES PARKINSONIANOS (Tese de doutorado do Braion, 2025)

	- Métodos para estimação/predição de tremor/movimento voluntário
		- Filtros g-h:
			- 1962 - Synthesis of an Optimal Set of Radar Track-While-Scan Smoothing Equations (BBF)
			- 1998 - Tracking and Kalman filtering made easy (CDF, BBF, g-h de maneira geral)
			- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data (CDF, BBF)
			- 2013 - A neuroprosthesis for tremor management through the control of muscle co-contraction (CDF)
		- FLC:
			- 1989 - Adaptive Fourier estimation of Time-Varying Evoked Potentials (FLC)
		- WFLC:
			- 1995 - Adaptive human-machine interface for persons with tremor 
			- 1996 - Modeling and Canceling Tremor in Human-Machine Interfaces
			- 1997 - Adaptive Fourier modeling for quantification of tremor
			- 1998 - Adaptive Canceling of Physiological Tremor for Improved Precision in Microsurgery 
			- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data (g-h + WFLC + KF)
			- 2014 - Adaptive sliding bandlimited multiple fourier linear combiner for estimation of pathological tremor (WFLC com BMFLC)
			- 2018 - Characterization of parkinsonian hand tremor and validation of a high-order tremor estimator (HWFLC-KF)
			- 2022 - Real-Time Performance Assessment of High-Order Tremor Estimators Used in a Wearable Tremor Suppression Device (compara HWFLC, WFLC, BMFLC)
			
		- Filtro digital/analógico:
			- 1999 - Tremor Suppression Using Functional Electrical Stimulation: A Comparison Between Digital and Analog Controllers
			- 2000 - Optimal digital filtering for tremor suppression (equalizador)
			- 2014 - Robust Controller for Tremor Suppression at Musculoskeletal Level in Human Wrist (passa-alta)
			- 2019 - Repetitive Control of Electrical Stimulation for Tremor Suppression (truncamento de filtro IIR)
			- 2020 - Algebraic estimator of Parkinson’s tremor frequency from biased and noisy sinusoidal signals
			
			**Obs.:** vários métodos baseados em FLC preparam dados com filtragem passa-alta / passa-banda.

		- Filtro adaptativo:
			- 2023 - Analysis of Adaptive Algorithms Based on Least Mean Square Applied to Hand Tremor Suppression Control
			- 2023 - A Real-Time Voluntary Motion Extraction Method Based on an Adaptive Filter
			- 2025 - A smart pen prototype with adaptive algorithms for stabilizing handwriting tremor signals in Parkinson’s disease (LMS, KF)
		
		- BMFLC:
			- 2007 - Bandlimited Multiple Fourier Linear Combiner for Real-time Tremor Compensation
			- 2010 - Estimation and filtering of physiological tremor for real-time compensation in surgical robotics applications
			- 2011 - Estimation of Physiological Tremor from Accelerometers for Real-Time Applications (BMFLC-KF, BMFLC-RLS)
			- 2014 - Adaptive sliding bandlimited multiple fourier linear combiner for estimation of pathological tremor (WFLC com BMFLC)
			- 2016 - Characterization of Upper-Limb Pathological Tremors - Application to Design of an Augmented Haptic Rehabilitation System (E-BMFLC)
			- 2016 - Prediction of pathological tremor using adaptive multiple oscillators linear combiner (refina estimativa do BMFLC com osciladores de Hopf)
			- 2024 - Tremor Estimation and Removal in Robot-Assisted Surgery Using Improved Enhanced Band-Limited Multiple Fourier Linear Combiner (IE-BMFLC)
		- FFT based:
			- 2016 - Estimation of Tremor Parameters and Extraction Tremor from Recorded Signals for Tremor Suppression
		- ABPF:
			- 2010 - Adaptive band-pass filter (ABPF) for tremor extraction from inertial sensor data
		- Hilbert-Huang transform based:
			- 2011 - Hilbert-Huang-Based Tremor Removal to Assess Postural Properties From Accelerometers
		- Autoregressivo:
			- 2013 - Physiological Tremor Estimation with Autoregressive (AR) Model and Kalman Filter for Robotics Applications (AR-KF, AR-LMS)
			- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications (AR-KF, AR-LMS)
		- Filtro de Kalman (KF):
			- 2008 - Kalman Filtering of Accelerometer and Electromyography (EMG) Data in Pathological Tremor Sensing System (KF)
			- 2010 - Real-Time Estimation of Pathological Tremor Parameters from Gyroscope Data (KF)
			- 2011 - Estimation of Physiological Tremor from Accelerometers for Real-Time Applications (BMFLC-KF, AR-KF)
			- 2013 - Physiological Tremor Estimation with Autoregressive (AR) Model and Kalman Filter for Robotics Applications (AR-KF)
			- 2013 - Multistep Prediction of Physiological Tremor for Surgical Robotics Applications (AR-KF)
			- 2016 - A zero phase adaptive fuzzy Kalman filter for physiological tremor suppression in robotically assisted minimally invasive surgery (ZPAFKF)
			- 2018 - Characterization of parkinsonian hand tremor and validation of a high-order tremor estimator (HWFLC-KF)
			- 2019 - Tremor Estimation and Removal in Robot-Assisted Surgery Using Lie Groups and EKF (EKF)
			- 2019 - WAKE: Wavelet decomposition coupled with adaptive Kalman filtering for pathological tremor extraction (Wavelet + KF)
			- 2025 - A smart pen prototype with adaptive algorithms for stabilizing handwriting tremor signals in Parkinson’s disease (LMS, KF)
		- AFE (tecnicamente um estimador de frequência):
			- 2019 - Adaptive notch filter for pathological tremor suppression using permanent magnet linear motor
		- Machine Learning / Deep Learning:
			- 2012 - Neural network based prediction of parkinsonian hand tremor using surface electromyography (Time Delayed FNN)
			- 2015 - Multistep prediction of physiological tremor based on machine learning for robotics assisted microsurgery (SVM)
			- 2019 - Training of Deep Bidirectional RNNs for Hand Motion Filtering via Multimodal Data Fusion (RNN)
			- 2020 - Quaternion broad learning system: A novel multi-dimensional filter for estimation and elimination tremor in teleoperation (Autoencoder)
			- 2020 - Three-domain fuzzy wavelet broad learning system for tremor estimation (Autoencoder)
			- 2020 - Prediction of Voluntary Motion Using Decomposition-and-Ensemble Framework With Deep Neural Networks (FNN)
			- 2021 - Real-Time Voluntary Motion Prediction and Parkinson's Tremor Reduction Using Deep Neural Networks (CNN, FNN)
			- 2021 - Exploring data-driven modeling and analysis of nonlinear pathological tremors (LSTM, FNN)
			- 2022 - Prediction of Pathological Tremor Signals Using Long Short-Term Memory Neural Networks (LSTM)
			- 2025 - Automatic differentiation of voluntary and tremulous motion using ensemble empirical mode decomposition and convolutional Bi-directional LSTM
			- 2025 - Parkinson’s disease tremor prediction towards real-time suppression: A self-attention deep temporal convolutional network approach (TCN)
			- 2026 - Tremor estimation and filtering in robotic-assisted surgery (SVM)

	- Métodos para supressão de tremor usando controle ativo (mas não necessariamente prótese)
		- 2013 - A neuroprosthesis for tremor management through the control of muscle co-contraction
		- 2014 - Robust Controller for Tremor Suppression at Musculoskeletal Level in Human Wrist
		- 2019 - Parkinson’s Tremor Suppression Using Active Vibration Control Method (PID)
		- 2019 - Adaptive notch filter for pathological tremor suppression using permanent magnet linear motor
		- 2019 - Repetitive Control of Electrical Stimulation for Tremor Suppression
		- 2024 - A Novel Approach to Parkinson’s Tremor Suppression: E-BMFLC and LADRC Integration ("artigo do access")
