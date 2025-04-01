export interface AspectPolarity {
  aspect: string;
  polarity: string;
}

export interface AnalysisResponse {
  aspects: AspectPolarity[];
}