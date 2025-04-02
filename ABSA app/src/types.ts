export interface AspectPolarity {
  aspect: string;
  polarity: string;
}

export interface AnalysisResponse {
  aspects: Array<{
    aspect: string;
    polarity: string;
  }>;
}