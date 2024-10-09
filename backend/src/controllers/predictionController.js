export const getPredictedPrice= async (req, res)=>{
    try{
        const predictedPrice= 299.99;
    res.json({predictedPrice});
    }
    catch(err){
        res.status(500).json({err:"Failed to get the predicted price."})
    }
}
