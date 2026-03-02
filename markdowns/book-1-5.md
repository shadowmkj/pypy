## COMPUTER GRAPHICS

C VERSION

<!-- image -->

DONALD HEARN • M. PAULINE BAKER

Contents

PREFACE

1

1-1

1-2

1-3

1-4

1-5

1-6

1-7

1-8

2

2-1

## Contents

Computer-Aided Design

Presentation Graphics xvii

2

4

11

2-2

2-3

Stereoscopic and Virtual-Reality

Systems

Raster-Scan Systems

Video Controller

Raster-Scan Display Processor

Random-Scan Systems

50

53

55

56

| PREFACE   | PREFACE                           | xvii   |     | Stereoscopicand Virtual-Reality Systems                             |
|-----------|-----------------------------------|--------|-----|---------------------------------------------------------------------|
| 1         | A Survey of Computer Graphics     | 2      | 2-2 | Raster-Scan System!; Video Controller Raster-Scan Display Processor |
|           | Computer-Aided Design             |        | 2-3 | Random-Scan Systems                                                 |
|           | Presentation Graphics             | 'I     | 2-4 | Graphics Monitors and Workstations                                  |
|           | Computer Art                      | l 3    | 2-5 | Input Devices                                                       |
|           | Entertainment                     | 18     |     | Keyboards                                                           |
|           | Education and Training            | 21     |     | Mouse                                                               |
|           | Visualization                     | 25     |     | Trackballand Spaceball                                              |
|           | Image Processing                  | 32     |     | Joysticks                                                           |
|           | Graphical User Interfaces         | 34     |     | Data Glove                                                          |
|           |                                   |        |     | Digitizers                                                          |
|           |                                   |        |     | Image Scanners                                                      |
|           |                                   |        |     | Touch Panels                                                        |
| 2         | Overview of Graphics systems      | 35     | 2-6 | Voice Systems Hard-Copy Devices                                     |
| 2-1       | VideoDisplayDevices               | 36     | 2-7 | Graphics Software                                                   |
|           | Refresh Cathode-Ray Tubes         | 37     |     | Coordinate Representations                                          |
|           | Raster-Scan Displays              | 40     |     | Graphics Functions                                                  |
|           | Random-Scan Displays              | 41     |     | SoftwareStandards                                                   |
|           | Color CRTMonitors                 | 42     |     | PHIGSWorkstations                                                   |
|           | Direct-View Storage Tubes         | 4.5    |     | Summary                                                             |
|           | Flat-Panel Displays               | 45     |     | References                                                          |
|           | Three-Dimensional Viewing Devices | 49     |     | Exercises                                                           |

| 3   |                                                       |    | Summary                              |     |
|-----|-------------------------------------------------------|----|--------------------------------------|-----|
|     | Outout Primitives                                     | 83 | Applications References              |     |
|     | Points and Lines                                      |    | Exercises                            |     |
|     | Line-DrawingAlgorithms DDAAlgorithm                   |    |                                      |     |
|     | Bresenham's Line Algorithm Parallel Line Algorithms   |    |                                      |     |
|     |                                                       |    | Attributes of Output                 |     |
|     | Loading the Frame Buffer                              |    | Primitives                           | 143 |
|     | Line Function                                         |    |                                      |     |
|     | Circle-GeneratingAlgorithms                           |    | Line Attributes                      |     |
|     | Properties of Circles                                 |    | LineType                             |     |
|     | Midpoint Circle Algorithm                             |    | LineWidth                            |     |
|     | Ellipse-GeneratingAlgorithms                          |    | Pen and Brush Options                |     |
|     | Properties of Ellipses                                |    | LineColor                            |     |
|     | Midpoint Ellipse Algorithm                            |    | Curve Attributes                     |     |
|     | Other Curves                                          |    | Color and Grayscale Levels           |     |
|     | Conic Sections                                        |    | Color Tables                         |     |
|     | Polynomials and SplineCurves                          |    | Grayscale                            |     |
|     | Parallel Curve Algorithms                             |    | Area-Fill Attributes                 |     |
|     | Curve Functions                                       |    | Fill Styles                          |     |
|     | Pixel Addressing                                      |    | Pattern Fill                         |     |
|     | and Object Geometry                                   |    | SoftFill                             |     |
|     | Screen Grid Coordinates                               |    | Character Attributes                 |     |
|     | Maintaining Geometric Properties of Displayed Objects |    | Text Attributes Marker Attributes    |     |
|     | Filled-Area Primitives                                |    | Bundled Attributes                   |     |
|     | Scan-LinePolygon F i l l Algorithm                    |    | Bundled Line Attributes              |     |
|     | Inside-Outside Tests                                  |    | Bundled Area-Fi Attributes           |     |
|     | Scan-Line Fill of Curved Boundary Areas               |    | Bundled Text Attributes              |     |
|     | Boundary-Fill Algorithm                               |    | Bundled Marker Attributes            |     |
|     |                                                       |    | Inquiry Functions                    |     |
|     | Flood -Fill Algorithm                                 |    | Antialiasing                         |     |
|     | Fill-Area Functions Cell Array                        |    | Supersampling Straight Line Segments |     |
|     | Character Generation                                  |    | Pixel-Weighting Masks                |     |

5

5-1

5-2

5-3

5-4

5-5

Area Sampling Straight Line

Segments

Filtering Techniques

Pixel Phasing

Summary

Exercises

Rotations

Shear

Systems

174

174

175

Contents

Affine Transformations

Transformation Functions

Summary

<!-- image -->

208

208

210

212

213

|     | Area Sampling Straight Line                                   |      | 5-6   | Aff ine Transformations                             | 208   |
|-----|---------------------------------------------------------------|------|-------|-----------------------------------------------------|-------|
|     | Segments                                                      | 174  | 5-7   | Transformation Functions                            | 208   |
|     | Filtering Techniques                                          | 174  | 5-8   | Raster Methods for Transformations                  | 210   |
|     | Pixel Phasing                                                 | 1 75 |       | Summary                                             | 212   |
|     | Compensating for Line lntensity                               |      |       | References                                          | 213   |
|     | Differences                                                   | 1 75 |       |                                                     |       |
|     | Antialiasing Area Boundaries                                  | 176  |       | Exercises                                           | 213   |
|     | Summary                                                       |      |       |                                                     |       |
|     | References                                                    |      | 6     | Two-Dimensional                                     |       |
|     | Exercises                                                     | 180  |       | Viewing                                             | 21 6  |
|     |                                                               |      | 6-1   | The Viewing Pipeline                                |       |
|     | Two-DimensionalGeometric                                      |      | 6-2   | Viewing Coordinate Reference Frame                  |       |
| 5   | Transformations                                               | 183  | 6-3   | Window-teviewport Coordinate Transformation         |       |
| 5-1 | Basic Transformations Translation                             |      |       | Two-DimensionalWewing Functions Clipping Operations |       |
|     | Rotation                                                      |      |       | Point Clipping                                      |       |
|     | Scaling                                                       |      |       | Line Clipping                                       |       |
| 5-2 | Matrix Representations                                        |      |       | Cohen-Sutherland Line Clipping                      |       |
|     | and Homogeneous Coordinates                                   |      |       | Liang-Barsky Line Clipping                          |       |
| 5-3 | Composite Transformations                                     |      |       | Nicholl-Lee-Nicholl Line Clipping                   |       |
|     | Translations                                                  |      |       | Line Clipping Using Nonrectangular Clip Windows     |       |
|     | Rotations                                                     |      |       | Splitting Concave Polygons                          |       |
|     | Scalings                                                      |      |       |                                                     |       |
|     | General Pivot-Point Rotation                                  |      |       | Polygon Clipping Sutherland-Hodgernan               |       |
|     | General Fixed-Point Scaling                                   |      |       | Polygon Clipping                                    |       |
|     | General Scaling Directions                                    |      |       | Weiler-Atherton Polygon Clipping                    |       |
|     | Concatenation Properties                                      |      |       | Other Polygon-Clipping Algorithms                   |       |
|     | General CompositeTransformations and Computational Efficiency |      |       | Curve Clipping                                      |       |
| 5-4 | Other Transformations                                         |      |       | Text Clipping                                       |       |
|     | Reflection Shear                                              |      |       | Exterior Clipping Summary                           |       |
| 5-5 | TransformationsBetween Coordinate                             | 205  |       | References                                          |       |
|     | Systems                                                       |      |       | Exercises                                           |       |

5-6

5-7

5-8

210

6

7

7-1

7-2

7-3

7-4

8

8-1

x

Contents

Structures and Hierarchical

Modeling

Structure Concepts

250

250

Accommodating Multiple

Skill Levels

Consistency

Backup and Error Handling

273

274

274

274

| 7   | Structures and Hierarchical                    | Structures and Hierarchical   | Structures and Hierarchical   | Accommodating Multiple                                  | Accommodating Multiple   |
|-----|------------------------------------------------|-------------------------------|-------------------------------|---------------------------------------------------------|--------------------------|
|     | Modeling                                       | 250                           |                               | Skill Levels Consistency                                |                          |
| 7-1 | Structure Concepts                             | 250                           |                               | Minimizing Memorization                                 |                          |
|     | Basic Structure Functions                      | 250                           |                               | Backup and Error Handling                               |                          |
|     | Setting Structure Attributes                   | 253                           |                               | Feedback                                                |                          |
| 7-2 | Editing Structures                             | 254                           | 8-2                           | lnput of Graphical Data                                 |                          |
|     | Structure Lists and the Element Pointer        | 255                           |                               | Logical Classification of Input Devices                 |                          |
|     | Setting the Edit Mode                          | 250                           |                               | Locator Devices                                         |                          |
|     | Inserting Structure Elements                   | 256                           |                               | Stroke Devices                                          |                          |
|     | Replacing Structure Elements                   | 257                           |                               | String Devices                                          |                          |
|     | Deleting Structure Elements                    | 257                           |                               | Valuator Devices                                        |                          |
|     | LabelingStructure Elements                     | 258                           |                               | Choice Devices                                          |                          |
|     | Copying Elements fromOneStructure              |                               |                               | Pick Devices                                            |                          |
|     | to Another                                     | 260                           | 8-3                           | lnput Functions                                         |                          |
| 7-3 | Basic Modeling Concepts                        | 260                           |                               | Input Modes                                             |                          |
|     | Mode1Representations                           | 261                           |                               | Request Mode                                            |                          |
|     | Symbol Hierarchies                             | 262                           |                               | Locator and Stroke Input in Request Mode                |                          |
|     | Modeling Packages. Hierarchical Modeling       | 263                           |                               | String Input in Request Mode                            |                          |
| 7-4 | with Structures                                | 265                           |                               | Valuator Input in Request Mode                          |                          |
|     | Local Coordinates and Modeling Transformations | 265                           |                               | Choice lnput in Request Mode Pick Input in Request Mode |                          |
|     | Modeling Transformations                       | 266                           |                               | SampleMode                                              |                          |
|     | Structure Hierarchies                          | 266                           |                               | Event Mode                                              |                          |
|     | Summary                                        | 268                           |                               | Concurrent Use of Input Modes                           |                          |
|     | References                                     | 269                           | 8-4                           | Initial Values for Input-Device Parameters              |                          |
|     | Exercises                                      |                               |                               |                                                         |                          |
|     |                                                | 2 69                          |                               | lnteractive Picture-Construction Techniques             |                          |
|     | Graphical User Interfaces                      |                               | 8-5                           | Basic Positioning Methods                               |                          |
|     | and Interactive lnput                          |                               |                               | Constraints Grids                                       |                          |
| 8   | Methods                                        | 271                           |                               | Gravity Field                                           |                          |
|     |                                                |                               |                               | Rubber-BandMethods                                      |                          |
| 8-1 | The User Dialogue                              |                               |                               | Dragging                                                |                          |
|     | Windows and Icons                              |                               |                               | Painting and Drawing                                    |                          |

Minimizing Memorization

8-6

9

9-1

9-2

Packages

10

10-1

10-2

10-3

Virtual-Reality Environments

Summary

References

Exercises

Views

292

293

294

294

10-4

10-5

10-6

Contents

Superquadrics

Superellipse

Superellipsoid

Blobby Objects

312

312

313

314

315

| 8-6                            | Virtual-Reality Environments       | 292                                | 10-4   | Superquadrics                                         |
|--------------------------------|------------------------------------|------------------------------------|--------|-------------------------------------------------------|
|                                | Summary                            | 233                                |        | Superellipse                                          |
|                                | References                         | 294                                |        | Superellipsoid                                        |
|                                | Exercises                          | 294                                | 10-5   | Blobby Objects                                        |
|                                |                                    |                                    | 10-6   | Spline Representations                                |
|                                |                                    |                                    |        | Interpolation and Approximation Splines               |
| 9                              | Three-Dimensional Concepts         | 296                                |        | Parametric Continuity Conditions Geometric Continuity |
| 9-1                            | Three-DimensionalDisplay Methods   | Three-DimensionalDisplay Methods   |        | Conditions                                            |
|                                | Parallel Projection                | Parallel Projection                |        | SplineSpecifications                                  |
|                                | Perspective Projection             | Perspective Projection             |        | Cubic Spline Interpolation Methods                    |
|                                | Visible Line and Surface           | Visible Line and Surface           |        |                                                       |
|                                |                                    |                                    |        | Natural Cubic Splines                                 |
|                                | Identification                     | Identification                     |        | Hermite Interpolation                                 |
|                                | Surface Rendering                  | Surface Rendering                  |        | Cardinal Splines                                      |
|                                | Exploded and Cutaway Views         | Exploded and Cutaway Views         |        | Kochanek-BartelsSplines                               |
|                                | Three-Dimensional and Stereoscopic | Three-Dimensional and Stereoscopic |        | Bezier Curves and Surfaces                            |
|                                | Views                              | Views                              |        | Bezier Curves                                         |
| 9-2                            | Three-DimensionalGraphics          | 302                                |        | Properties of Bezier Curves                           |
|                                |                                    |                                    |        | Design Techniques Using Bezier Curves                 |
|                                |                                    |                                    |        | Cubic Ezier Curves                                    |
|                                | Three-Dimensional                  | Three-Dimensional                  |        | Bezier Surfaces                                       |
|                                |                                    |                                    |        | B-SplineCurves and Surfaces B-Spline Curves           |
|                                |                                    |                                    |        | Uniform, Periodic B-Splines                           |
|                                |                                    |                                    |        | Cubic, Periodic €3-Splines                            |
| 10-1 Polygon Surfaces          | 10-1 Polygon Surfaces              | 10-1 Polygon Surfaces              |        | Open, Uniform B-Splines                               |
| Polygon Tables                 | Polygon Tables                     | Polygon Tables                     |        | Nonuniform 13-Splines                                 |
| Plane Equations                | Plane Equations                    | Plane Equations                    |        | B-SplineSurfaces                                      |
| Polygon Meshes                 | Polygon Meshes                     | Polygon Meshes                     |        | Beta-Splines                                          |
| 10-2 Curved Lines and Surfaces | 10-2 Curved Lines and Surfaces     | 10-2 Curved Lines and Surfaces     |        | Beta-SplineContinuity                                 |
| 10-3 Quadric Sutiaces          | 10-3 Quadric Sutiaces              | 10-3 Quadric Sutiaces              |        | Conditions                                            |
| Sphere                         | Sphere                             | Sphere                             |        | Cubic, Periodic Beta-Spline                           |
| Ellipsoid                      | Ellipsoid                          | Ellipsoid                          |        | Matrix Representation                                 |
| Torus                          | Torus                              | Torus                              |        | Rational Splines                                      |