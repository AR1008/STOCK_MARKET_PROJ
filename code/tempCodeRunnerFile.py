def load_test_data(self):
        """Load and prepare test data"""
        logger.info("Loading test data...")
        
        try:
            # Load combined data (should exist from training)
            if os.path.exists('data/combined_data.csv'):
                df = pd.read_csv('data/combined_data.csv')
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                # Recreate combined data if it doesn't exist
                logger.info("Combined data not found, recreating...")
                df = self.recreate_combined_data()
            
            # Engineer features (same as training)
            df = self.engineer_features(df)
            
            # Prepare the same way as training
            X, y, dates = self.prepare_test_features(df)
            
            # Split data the same way as training (60/20/20)
            n_samples = len(X)
            train_end = int(n_samples * 0.6)
            val_end = int(n_samples * 0.8)
            
            # We want the test set
            X_test = X.iloc[val_end:].copy()
            y_test = y[val_end:]
            test_dates = dates[val_end:]
            
            logger.info(f"Test data prepared: {len(X_test)} samples, {len(X_test.columns)} features")
            return X_test, y_test, test_dates, df
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise