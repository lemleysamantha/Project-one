-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "Vehicles" (
    "id" int   NOT NULL,
    "price" int   NOT NULL,
    "year" date   NOT NULL,
    "manufacturer" varchar   NOT NULL,
    "model" varchar   NOT NULL,
    "condition" varchar   NOT NULL,
    "cylinders" int   NOT NULL,
    "fuel" varchar   NOT NULL,
    "odometer" int   NOT NULL,
    "title_status" varchar   NOT NULL,
    "transmission" varchar   NOT NULL,
    "drive" varchar   NOT NULL,
    "type" varchar   NOT NULL,
    CONSTRAINT "pk_Vehicles" PRIMARY KEY (
        "id"
     )
);

