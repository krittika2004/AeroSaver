import express from 'express';
import bodyParser from 'body-parser';
import predictionRoutes from './routes/predictionRoutes.js';
const app = express();
const PORT = process.env.PORT || 3000;
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
app.use("/api/predictions", predictionRoutes());
app.get("/", (req, res)=>{
    res.send("This is home page");
})
app.listen(PORT, ()=>{
    console.log(`Server is listening on port ${PORT}`);
})
