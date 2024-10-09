import express from 'express';
import { getPredictedPrice } from '../controllers/predictionController.js';
const predictionRoutes=()=>{
    const router=express.Router();
    router.get("/predict", getPredictedPrice );
    return router;

}
export default predictionRoutes;