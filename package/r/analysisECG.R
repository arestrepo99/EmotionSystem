# Function analysisECG returns ECG features 
analysisECG <- function(ecg, fs, t, gr_r, gr2, gr3, gr4, gr5, gr6, gr7, gr8, gr9, gr10) {
  
  t = unlist(t)

  # low-pass filter with cut-off frequency of 100 Hz
  library(signal)
  f <- butter(3, 100/(fs/2), "low")
  ecg <- filtfilt(f,ecg)
  
  # high-pass filter with cut-off frequency of 1 Hz 
  f1 <- butter(3,1/(fs/2), "high")
  ecg <- filtfilt(f1, ecg)
  
  # normalization 
  ecg <- (ecg-min(ecg))/(max(ecg)-min(ecg))
  
  ### PQRST detection
  # # R peak detection
  library(pracma)
  pikovi <- findpeaks(ecg, minpeakheight = gr_r, minpeakdistance = 600)
  locs_R <- pikovi[,2] + min(t)*fs
  locs_R <- sort(locs_R)
  
  #### averaging ecg signal
  i <- 1
  segment <- 0.75*fs # length of each segment which will be extracted is 750 ms
  N <- length(locs_R) # number of peaks in the signal
  N1 <- 10 # number of PQRST segments of ECG signal, which will be used for averaging
  N2 <- 2 #  averaging starts from second beat (first beat maybe does not consists whole PQRST segment)
  PQRST <- matrix(0, segment, N1)
  ECG1 <- 0  # 10 PQRST intervals will be saved here
  ECG <- matrix(0,segment,(N-N1+1))
  m <- 1
  k <- 1
  while (k < (N-N1+1)) {
    for (i in 1:N1) {
      PQRST[,i] <- ecg[(locs_R[N2 + i - 1] - 0.25*fs + 1):(locs_R[N2 + i - 1] + 0.5*fs)]
      ECG1 <- ECG1 + PQRST[,i]
      i <- i + 1
    }
    ECG1 <- ECG1/N1
    ECG[,m] <- ECG1
    ECG1 <- 0
    PQRST <- matrix(0, segment, N1)
    N2 <- N2 + 1
    m <- m + 1
    k <- k + 1
  }
  
  ecg_average <- as.vector(ECG)
  t2 <- (1:length(ecg_average))/fs
  
  ########################################
  # detection of R peak on ecg_average
  library(pracma)
  pikovi_av <- findpeaks(ecg_average, minpeakheight = gr_r, minpeakdistance = 600)
  locs_Rav <- pikovi_av[,2] + min(t2)*fs
  locs_Rav <- sort(locs_Rav)
  
  # S peak detection
  locs_S <- c()                                                                   
  for (kk in 1:(length(locs_Rav))) {
    ocs <- which.min(ecg_average[locs_Rav[kk] : (locs_Rav[kk] + gr2)])
    locs_S[kk] <- ocs + locs_Rav[kk]
  }
  
  
  # S2 detection
  locs_S2 <- c()
  for (kk in 1:(length(locs_S)-1)) {
    ocs <- which.max(ecg_average[locs_S[kk] : (locs_S[kk] + gr10)])
    locs_S2[kk] <- ocs + locs_S[kk]
  }
  
  # T peak detection
  locs_T <- c()
  for (kk in 1:(length(locs_S)-1)) {
    ocs <- which.max(ecg_average[locs_S[kk] : (locs_S[kk] + gr3)])
    locs_T[kk] <- ocs + locs_S[kk]
  }
  
  # Q peak detection
  locs_Q <- c()
  for (kk in 1:(length(locs_Rav))) {
    ocs <- which.min(ecg_average[locs_Rav[kk] : (locs_Rav[kk] - gr4)])
    locs_Q[kk] <- locs_Rav[kk] - ocs
  }
  
  # P peak detection
  locs_P <- c()
  for (kk in 1:(length(locs_Q))) {
    ocs <- which.max(ecg_average[locs_Q[kk] : (locs_Q[kk] - gr5)])
    locs_P[kk] <- locs_Q[kk] - ocs
  }
  
  ### Detection of the beginning and the end of P and T wave
  
  # P1 detection (beginning of P wave)
  locs_P1 <- c()
  for (kk in 1:(length(locs_P))) {
    ocs <- which.min(ecg_average[locs_P[kk] : (locs_P[kk] - gr6)])
    locs_P1[kk] <- locs_P[kk] - ocs
  }
  
  # P2 detection (end of P wave)
  locs_P2 <- c()
  for (kk in 1:(length(locs_P))) {
    ocs <- which.min(ecg_average[locs_P[kk] : (locs_P[kk] + gr7)]) 
    locs_P2[kk] <- ocs + locs_P[kk]
  }
  
  # T1 detection (beginning of T wave)
  locs_T1 <- c()
  for (kk in 1:(length(locs_T))) {
    ocs <- which.min(ecg_average[locs_T[kk] : (locs_T[kk] - gr8)]) 
    locs_T1[kk] <- locs_T[kk] - ocs
  }
  
  # T2 detection (end of T wave)
  locs_T2 <- c()
  for (kk in 1:(length(locs_T))) {
    ocs <- which.min(ecg_average[locs_T[kk] : (locs_T[kk] + gr9)])
    locs_T2[kk] <- ocs + locs_T[kk]
  }
  
  start <- 0
  end <- 5
  plot(t2, ecg_average, type = "l", xlim = c(start,end), xlab = "Time [s]", ylab = "Amplitude [mV]", main = "ECG")
      grid()
    points(locs_P/fs, ecg_average[locs_P], type = "p", col = "black", pch = 16, xlim = c(start,end))
    points(locs_Q/fs, ecg_average[locs_Q], type = "p", col = "black", pch = 15, xlim = c(start,end))
    points(locs_Rav/fs, ecg_average[locs_Rav], type = "p", col = "black", pch = 17, xlim = c(start,end))
    points(locs_S/fs, ecg_average[locs_S], type = "p", col = "black", pch = 18, xlim = c(start,end))
    points(locs_T/fs, ecg_average[locs_T], type = "p", col = "black", pch = 8, xlim = c(start,end))
    points(locs_T1/fs, ecg_average[locs_T1], type = "p", col = "black", pch = 7, xlim = c(start,end))
    points(locs_T2/fs, ecg_average[locs_T2], type = "p", col = "black", pch = 6, xlim = c(start,end))
    points(locs_P1/fs, ecg_average[locs_P1], type = "p", col = "black", pch = 5, xlim = c(start,end))
    points(locs_P2/fs, ecg_average[locs_P2], type = "p", col = "black", pch = 4, xlim = c(start,end))
    points(locs_S2/fs, ecg_average[locs_S2], type = "p", col = "black", pch = 18, xlim = c(start,end))
      legend("topright", c("P", "Q", "R", "S", "T", "T1", "T2", "P1", "P2", "S2"), 
            col = c("black", "black", "black", "black", "black", "black", "black", "black", "black", "black"), 
            pch = c(16,15,17,18,8,7,6,5,4,18), cex = 0.6)
  #...............................................................................................................................
  ### frequency domain
  ecg_fft <- fft(ecg)
  ecg_fft_magn <- Mod(ecg_fft)
  ecg_fft_magn <- ecg_fft_magn[1:(length(ecg_fft_magn)/2)]
  fosa <- 1:length(ecg_fft_magn)/max(t)
  
  plot(fosa, ecg_fft_magn, type = "l", xlab = "frequency [Hz]", ylab = "magnitude [a.u.]", main = "FFT", xlim = c(0,50))
      grid()
  
  ### HRV
  RR <- vector("numeric")
  for (ind in 1:(length(locs_R)-1)) {
    RR[ind] <- locs_R[ind+1]/fs - locs_R[ind]/fs
  }
  
  HR <- 1/RR*60
  HR_mean <- mean(HR) # beats per minute
  RR_mean <- mean(RR)
  
  t_RR <- locs_R[2:length(locs_R)]/fs # this is not equidistant
  plot(t_RR, RR, type = "l", xlab = "vreme [s]", ylab = "RR interval [s]",
       main = "HRV")
      grid()
  points(t_RR, RR, type = "p", pch = 20, col = "blue")
  
  plot(t_RR, HR, type = "l", xlab = "vreme [s]", ylab = "HR [bpm]")
      grid()
  points(t_RR, HR, type = "p", pch = 20, col = "blue")
  
  #....................................................................................................................
  ### HRV based features - time domain ###
  pom <- vector("numeric")
  for (ind in 1:(length(RR)-1)) {
    pom[ind] <- (RR[ind] - RR[ind+1])^2
  }
  
  rmssd <- sqrt(sum(pom)/length(pom)) # in seconds
  sdnn <- sd(RR) # in seconds
  mn_nn <- mean(RR) # in seconds
  m_nn <- max(RR) # in seconds
  
  razlika <- vector("numeric")
  for (ind in 1:(length(RR)-1)) {
    razlika[ind] <- abs(RR[ind] - RR[ind+1])
  }
  nn50 <- sum(razlika > 0.05) # number of pairs of adjacent RR intervals differing by more than 50 ms in the entire recording
  pnn50 <- (nn50/length(RR))*100 # %
  sdsd <- sd(razlika) # standard deviation of differences between adjacent NN intervals
  
  # HRV index
  width <- seq(min(RR), max(RR), 1/fs)
  h <- hist(RR, breaks = width)
  y <- max(h$density)
  HRV_index <- length(RR)/y # integral of the density distribution (the number of all RR intervals)
  # divided by the maximum of the density distribution (bins of 1/fs s)
  
  #..................................................................................................................
  ### HRV based features - frequency ### 
  ### FFT HRV-a
  t_RR <- locs_R[2:length(locs_R)]/fs # this is not equidistant
  
  interpolacija <- spline(t_RR, RR, xmin = min(t_RR), xmax = max(t_RR))
  par(mfrow = c(2,1))
  plot(t_RR, RR, type = "l", xlab = "vreme [s]", ylab = "RR interval [s]",
       main = "HRV")
      grid()
  points(t_RR, RR, type = "p", pch = 20, col = "blue")
  plot(interpolacija$x, interpolacija$y, type = "l",
       main = "HRV - interpolacija RR intervala", xlab = "vreme [s]", ylab = "RR interval [s]") #inter$x is equdistant with the step of 0.2629621
  points(interpolacija$x, interpolacija$y, type = "p", pch = 20, col = "blue")
      grid()
  
  rr_fft <- fft(interpolacija$y)
  rr_fft_magn <- Mod(rr_fft)
  rr_fft_magn <- rr_fft_magn[1:(length(rr_fft_magn)/2)]
  fosa2 <- 1:length(rr_fft_magn)/max(interpolacija$x)
  
  psd_rr <- rr_fft_magn ^ 2 # power spectral density
  
  plot(fosa2, psd_rr, type = "l", xlab = "frekvencija [Hz]", ylab = "magnituda", 
       main = "PSD RR intervala", xlim = c(0.04,0.4), ylim = c(0.01, 30))
  grid()
  
  LF <- c(0.04, 0.15) # low frequency
  HF <- c(0.15, 0.4) # high frequency
  VLF <- c(0, 0.04) # very low frequency
  
  iLF = (fosa2 >= LF[1]) & (fosa2 <= LF[2])
  iHF = (fosa2 >= HF[1]) & (fosa2 <= HF[2])
  
  aLF <- trapz(fosa2[iLF],psd_rr[iLF]) # area under the curve in a given range - LF spectral power 
  aHF <- trapz(fosa2[iHF],psd_rr[iHF]) # area under the curve in a given range - HF spectral power
  nesto <- psd_rr[iLF]
  LFHF <- aLF/aHF

  ###########################################################################
  
  iTotalPower <- (fosa2 >= VLF[1]) & (fosa2 <= HF[2])
  total_power <- trapz(fosa2[iTotalPower], psd_rr[iTotalPower])
  
  iVLF <- (fosa2 >= VLF[1]) & (fosa2 <= VLF[2])
  aVLF <- trapz(fosa2[iVLF],psd_rr[iVLF]) # VLF spectral power
  lfnu <- aLF/(total_power - aVLF)*100 # low-frequency power in normalized units
  hfnu <- aHF/(total_power - aVLF)*100 # high-frequency power in normalized units
  
  
  #.............................................................................................................................
  ### HRV based features - geometric ###
  rr1 <- RR[2:length(RR)]
  rr2 <- RR[1:(length(RR)-1)]
  linline <- lm(rr1~rr2)
  par(mfrow = c(1,1))
  plot(rr1, rr2, xlab = "RR_N [s]", ylab = "RR_N-1 [s]", main = "Poincare plot", pch = 20)
      grid()
  abline(linline, lty = "longdash")
  
  SD1 <- sd(rr2-rr1)/sqrt(2) # geometric deviations between consecutive R-R
  SD2 <- sd(rr2+rr1)/sqrt(2) 
  SDRR <- sqrt(SD1^2 + SD2^2)/sqrt(2) #same parameter as the sdnn 
  
  #............................................................................................................................
  ### With-in beat features of the ECG signal ###
  # statistical features from each interval: min, max, median, mean, standard deviation
  
  #PR distance
  PR <- vector("numeric")
  for (ind in 1:ifelse(length(locs_P)>length(locs_Rav), length(locs_Rav), length(locs_P))) {
    PR[ind] <- locs_Rav[ind]/fs - locs_P[ind]/fs
  }
  
  min_pr <- min(PR, na.rm = TRUE)
  max_pr <- max(PR, na.rm = TRUE)
  mean_pr <- mean(PR, na.rm = TRUE)
  median_pr <- median(PR, na.rm = TRUE)
  sd_pr <- sd(PR, na.rm = TRUE)
  
  # ST distance
  ST <- vector("numeric")
  for (ind in 1:ifelse(length(locs_S)>length(locs_T), length(locs_T), length(locs_S))) {
    ST[ind] <- locs_T[ind]/fs - locs_S[ind]/fs
  }
  
  min_st <- min(ST, na.rm = TRUE)
  max_st <- max(ST, na.rm = TRUE)
  mean_st <- mean(ST, na.rm = TRUE)
  median_st <- median(ST, na.rm = TRUE)
  sd_st <- sd(ST, na.rm = TRUE)
  
  # QRS distance
  QRS <- vector("numeric")
  for (ind in 1: ifelse(length(locs_Q)>length(locs_S), length(locs_S), length(locs_Q))) {
    QRS[ind] <- locs_S[ind]/fs - locs_Q[ind]/fs
  }
  
  min_qrs <- min(QRS, na.rm = TRUE)
  max_qrs <- max(QRS, na.rm = TRUE)
  mean_qrs <- mean(QRS, na.rm = TRUE)
  median_qrs <- median(QRS, na.rm = TRUE)
  sd_qrs <- sd(QRS, na.rm = TRUE)
  
  ### Clinical features 
  # PR interval (P1Q)
  PR_interval <- vector("numeric")
  for (ind in 1:ifelse(length(locs_P1)>length(locs_Q), length(locs_Q), length(locs_P1))) {
    PR_interval[ind] <- locs_Q[ind]/fs - locs_P1[ind]/fs
  }
  PR_interval_mean <- mean(PR_interval, na.rm = TRUE)
  PR_interval_sd <- sd(PR_interval, na.rm = TRUE)
  
  # PR segment (P2Q)
  PR_segment <- vector("numeric")
  for (ind in 1:ifelse(length(locs_P2)>length(locs_Q), length(locs_Q), length(locs_P2))) {
    PR_segment[ind] <- locs_Q[ind]/fs - locs_P2[ind]/fs
  }
  PR_segment_mean <- mean(PR_segment, na.rm = TRUE)
  PR_segment_sd <- sd(PR_segment, na.rm = TRUE)
  
  # ST interval (S2T2)
  ST_interval <- vector("numeric")
  for (ind in 1:ifelse(length(locs_S2)>length(locs_T2), length(locs_T2), length(locs_S2))) {
    ST_interval[ind] <- locs_T2[ind]/fs - locs_S2[ind]/fs
  }
  ST_interval_mean <- mean(ST_interval, na.rm = TRUE)
  ST_interval_sd <- sd(ST_interval, na.rm = TRUE)
  
  # ST segment (S2T1)
  ST_segment <- vector("numeric")
  for (ind in 1:ifelse(length(locs_S2)>length(locs_T1), length(locs_T1), length(locs_S2))) {
    ST_segment[ind] <- locs_T1[ind]/fs - locs_S2[ind]/fs
  }
  ST_segment_mean <- mean(ST_segment, na.rm = TRUE)
  ST_segment_sd <- sd(ST_segment, na.rm = TRUE)
  
  # QT interval (QT2)
  QT_interval <- vector("numeric")
  for (ind in 1:ifelse(length(locs_Q)>length(locs_T2), length(locs_T2), length(locs_Q))) {
    QT_interval[ind] <- locs_T2[ind]/fs - locs_Q[ind]/fs
  }
  
  # QTnorm, Bazett's formula
  QT_norm <- QT_interval/sqrt(RR[1:length(QT_interval)])
  QT_norm_mean <- mean(QT_norm, na.rm = TRUE)
  QT_norm_sd <- sd(QT_norm, na.rm = TRUE)
  
  # QRS complex (QS2)
  QRS_complex <- vector("numeric")
  for (ind in 1: ifelse(length(locs_Q)>length(locs_S2), length(locs_S2), length(locs_Q))) {
    QRS_complex[ind] <- locs_S2[ind]/fs - locs_Q[ind]/fs
  }
  QRS_complex_mean <- mean(QRS_complex, na.rm = TRUE)
  QRS_complex_sd <- sd(QRS_complex, na.rm = TRUE)
  
  # T wave duration (T1T2)
  T_wave <- vector("numeric")
  for (ind in 1: ifelse(length(locs_T1)>length(locs_T2), length(locs_T2), length(locs_T1))) {
    T_wave[ind] <- locs_T2[ind]/fs - locs_T1[ind]/fs
  }
  T_wave_mean <- mean(T_wave, na.rm = TRUE)
  T_wave_sd <- sd(T_wave, na.rm = TRUE)
  
  # P wave duration (P1P2)
  P_wave <- vector("numeric")
  for (ind in 1: ifelse(length(locs_P1)>length(locs_P2), length(locs_P2), length(locs_P1))) {
    P_wave[ind] <- locs_P2[ind]/fs - locs_P1[ind]/fs
  }
  P_wave_mean <- mean(P_wave, na.rm = TRUE)
  P_wave_sd <- sd(P_wave, na.rm = TRUE)
  
  #..................................................................................................................
  ### Amplitude
  
  prA <- vector("numeric")
  for (ind in 1:ifelse(length(locs_P)>length(locs_Rav), length(locs_Rav), length(locs_P))) {
    prA[ind] <- ecg_average[locs_Rav[ind]] - ecg_average[locs_P[ind]]
  } 
  
  rqA <- vector("numeric")
  for (ind in 1:ifelse(length(locs_Q)>length(locs_Rav), length(locs_Rav), length(locs_Q))) {
    rqA[ind] <- abs(ecg_average[locs_Rav[ind]] - ecg_average[locs_Q[ind]])
  } 
  
  rsA <- vector("numeric")
  for (ind in 1:ifelse(length(locs_S)>length(locs_Rav), length(locs_Rav), length(locs_S))) {
    rsA[ind] <- abs(ecg_average[locs_Rav[ind]] - ecg_average[locs_S[ind]])
  } 
  
  rtA <- vector("numeric")
  for (ind in 1:ifelse(length(locs_T)>length(locs_Rav), length(locs_Rav), length(locs_T))) {
    rtA[ind] <- abs(ecg_average[locs_R[ind]] - ecg_average[locs_T[ind]])
  } 
  
  tsA <- vector("numeric")
  for (ind in 1:ifelse(length(locs_S)>length(locs_T), length(locs_T), length(locs_S))) {
    tsA[ind] <- abs(ecg_average[locs_T[ind]] - ecg_average[locs_S[ind]])
  } 
  
  qsA <- vector("numeric")
  for (ind in 1:ifelse(length(locs_S)>length(locs_Q), length(locs_Q), length(locs_S))) {
    qsA[ind] <- abs(ecg_average[locs_S[ind]] - ecg_average[locs_Q[ind]])
  } 
  
  prA_mean <- mean(prA, na.rm = TRUE)
  rqA_mean <- mean(rqA, na.rm = TRUE)
  rsA_mean <- mean(rsA, na.rm = TRUE)
  rtA_mean <- mean(rtA, na.rm = TRUE)
  tsA_mean <- mean(tsA, na.rm = TRUE)
  qsA_mean <- mean(qsA, na.rm = TRUE)
  
  prA_sd <- sd(prA, na.rm = TRUE)
  rqA_sd <- sd(rqA, na.rm = TRUE)
  rsA_sd <- sd(rsA, na.rm = TRUE)
  rtA_sd <- sd(rtA, na.rm = TRUE)
  tsA_sd <- sd(tsA, na.rm = TRUE)
  qsA_sd <- sd(qsA, na.rm = TRUE)
  
  # Clinical feature Ek (amplitude)
  # Ek = alpha*(Tampl*Rampl)/(RSampl)^2
  Tampl <- vector("numeric")
  for (ind in 1:ifelse(length(locs_T1)>length(locs_T), length(locs_T), length(locs_T1))) {
    Tampl[ind] <- abs(ecg_average[locs_T1[ind]] - ecg_average[locs_T[ind]])
  }
  
  Rampl <- vector("numeric")
  for (ind in 1:ifelse(length(locs_P1)>length(locs_Rav), length(locs_Rav), length(locs_P1))) {
    Rampl[ind] <- abs(ecg_average[locs_P1[ind]] - ecg_average[locs_Rav[ind]])
  }
  
  alpha <- 10
  Ek <- alpha*(Tampl*Rampl[1:length(Tampl)])/(rsA[1:length(Tampl)])^2
  Ek_mean <- mean(Ek, na.rm = TRUE)
  Ek_sd <- sd(Ek, na.rm = TRUE)
  
  obelezje <- c("HR_mean", "RR_mean", "rmssd", "sdnn", "mn_nn", "m_nn", "nn50", "pnn50", "sdsd", "HRV index",
                "lf", "hf", "lfhf", "lfnu", "hfnu", "total_power", "SD1", "SD2",
                "PR min", "PR max", "PR mean", "PR median", "PR sd",
                "ST min", "ST max", "ST mean", "ST median", "ST sd",
                "QRS min", "QRS max", "QRS mean", "QRS median", "QRS sd",
                "PR ampl", "RQ ampl", "RS ampl", "RT ampl", "ST ampl", "QS ampl",
                "PRa sd", "RQa sd", "RSa sd", "RTa sd", "STa sd", "QSa sd",
                "PRinterval_mean", "PRinterval_sd", "PRsegment_mean", "PRsegment_sd",
                "STinterval_mean", "STinterval_sd", "STsegment_mean", "STsegment_sd",
                "QRScomplex_mean", "QRScomplex_sd", "QTnorm_mean", "QTnorm_sd", 
                "Twave_mean", "Twave_sd", "Pwave_mean", "Pwave_sd", "Ek_mean", "Ek_sd")
  vrednost <- c(HR_mean, RR_mean, rmssd, sdnn, mn_nn, m_nn, nn50, pnn50, sdsd, HRV_index,
                aLF, aHF, LFHF, lfnu, hfnu, total_power, SD1, SD2,
                min_pr, max_pr, mean_pr, median_pr, sd_pr, min_st, max_st, mean_st, median_st, sd_st,
                min_qrs, max_qrs, mean_qrs, median_qrs, sd_qrs,
                prA_mean, rqA_mean, rsA_mean, rtA_mean, tsA_mean, qsA_mean,
                prA_sd, rqA_sd, rsA_sd, rtA_sd, tsA_sd, qsA_sd,
                PR_interval_mean, PR_interval_sd, PR_segment_mean, PR_segment_sd,
                ST_interval_mean, ST_interval_sd, ST_segment_mean, ST_segment_sd,
                QRS_complex_mean, QRS_complex_sd, QT_norm_mean, QT_norm_sd,
                T_wave_mean, T_wave_sd, P_wave_mean, P_wave_sd, Ek_mean, Ek_sd)
  veliki_frejm <- data.frame(obelezje, vrednost)
  
  
  return(veliki_frejm)
 }