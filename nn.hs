{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot #-}

import Numeric.LinearAlgebra
import System.Random

type Initializer g = g -> Int -> Int -> Matrix Double

type LossFunction = Vector Double -> Vector Double -> Double

data Activation = Activation
  { forward :: Vector Double -> Vector Double,
    derivative :: Vector Double -> Vector Double
  }

data Layer = Layer
  { weights :: Matrix Double,
    bias :: Matrix Double,
    activation :: Activation
  }

data LayerData g = LayerData
  { inSize :: Int,
    outSize :: Int,
    activation :: Activation,
    initializer :: Initializer g
  }

data ModelData g = ModelData
  { layers :: [LayerData g],
    loss :: LossFunction,
    inSize :: Int,
    outSize :: Int,
    gen :: g
  }

data Model = Model
  { layers :: [Layer],
    loss :: LossFunction,
    inSize :: Int,
    outSize :: Int
  }

pred' :: Model -> Vector Double -> Vector Double
pred' model x
  | size x == model.inSize =
      foldl
        (\z (y :: Layer) -> y.activation.forward (flatten y.bias + (y.weights #> z)))
        x
        model.layers
  | otherwise =
      error
        ( "Expected Vector of size "
            ++ show model.inSize
            ++ " but found size "
            ++ show (size x)
        )

-- Split one generator into n generators
splitN :: (RandomGen g) => Int -> g -> [g]
splitN n gen
  | n <= 0 = []
  | n == 1 = [gen]
  | otherwise =
      let (g1, g2) = split gen
       in g1 : splitN (n - 1) g2

splitInf :: (RandomGen g) => g -> [g]
splitInf g = g1 : splitInf g2
  where
    (g1, g2) = split g

createLayer :: (RandomGen g) => g -> LayerData g -> Layer
createLayer gen ld =
  Layer
    { weights = weights,
      bias = bias,
      activation = ld.activation
    }
  where
    weights = ld.initializer gen ld.outSize ld.inSize
    bias = ld.initializer gen ld.outSize 1

createModel :: (RandomGen g) => ModelData g -> Model
createModel md =
  Model
    { layers = layers,
      loss = md.loss,
      inSize = md.inSize,
      outSize = md.outSize
    }
  where
    layers =
      [ createLayer gen ld
        | (ld, gen) <- zip md.layers (splitInf md.gen)
      ]

conInit :: (RandomGen g) => Double -> Initializer g
conInit c _ rows cols = (rows >< cols) gen
  where
    gen = [c, c ..]

uniformInit ::
  (RandomGen g) =>
  Double ->
  Double ->
  Initializer g
uniformInit low high rgen rows cols =
  matrix cols (take (rows * cols) gen)
  where
    gen = randomRs (low, high) rgen

relu =
  Activation
    { forward = cmap (max 0),
      derivative = cmap (\x -> if x < 0 then 0 else 1)
    }

mse :: LossFunction
mse yTrue yPred = sumElements (cmap (^ 2) diff) / n
  where
    diff = yTrue - yPred
    n = fromIntegral (size yTrue)

defaultLayer :: (RandomGen g) => LayerData g
defaultLayer =
  LayerData
    { inSize = 3,
      outSize = 4,
      activation = relu,
      initializer = conInit 1
    }

m =
  createModel
    ( ModelData
        { inSize = 2,
          outSize = 8,
          gen = mkStdGen 2,
          layers =
            [ defaultLayer {inSize = 2, outSize = 4},
              defaultLayer {inSize = 4, outSize = 4}
            ],
          loss = mse
        }
    )

main = do
  _ <- print (head m.layers).weights
  _ <- print (m.layers !! 1).weights
  _ <- print (length m.layers)
  let p = pred' m (fromList [-2, 1])
  _ <- print p
  print "done"
