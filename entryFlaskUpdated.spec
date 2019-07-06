# -*- mode: python -*-

block_cipher = None


a = Analysis(['entryFlaskUpdated.py', 'Forecasting.py', 'FPGrowth.py', 'LassoRegression.py', 'GradientBoostingClassificationTest.py', 'GradientBoostingRegressionTest.py', 'LinearRegression.py', 'RandomForestClassifier.py', 'RidgeRegression.py', 'SentimentAnalysis.py'],
             pathex=['/home/fidel/DeepInsight/branch/DeepInsightAnalytics/src/PredictionAlgorithms'],
             binaries=[],
             datas=[],
             hiddenimports=['tkinter', 'scipy', 'cython', 'matplotlib'],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='entryFlaskUpdated',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='entryFlaskUpdated')
