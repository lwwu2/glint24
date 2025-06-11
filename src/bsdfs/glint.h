/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__GLINT_H)
#define __GLINT_H

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <boost/algorithm/string.hpp>

#include <stack>

#define GLINT_TREE_LEVEL 4

#define GLINT_UNBIASED


MTS_NAMESPACE_BEGIN

inline int sign(float x) { 
    int t = x<0 ? -1 : 0;
    return x > 0 ? 1 : t;
}


void clip_arc(Point2 p0, Point2 p1, const float c, Point3* p) {
    // p: [y0,0: not clip, 1: clip, -1: drop],[y1,0: not clip, 1: clip, -1: drop]
    const float x0=p0.x;
    const float y0=p0.y;
    const float x1=p1.x;
    const float y1=p1.y;
    
    float Dx = x1-x0;
    float Dy = y1-y0;
    
    float C = c*c*(y0*y0-1)+x0*x0;
    
    if (C>0 && x0<0) {// p0 needs to be clipped
        float A = (c*c*Dy*Dy+Dx*Dx);
        A = std::max(A,1e-12f);
        
        float B = 2*(c*c*y0*Dy+x0*Dx);
        float D = B*B-4*A*C;
        float sqrtD = sqrtf(std::max(D,0.0f));
        float t0 = (-B-sqrtD)/(2*A);
        float t1 = (-B+sqrtD)/(2*A);
        
        if (c*c*(y1*y1-1)+x1*x1>0 && x1<0) {// both need to be clipped
            if (D<0||t0<0||t0>1||t1<0||t1>1) {// no intersection
                p[0].z = -1;
                p[1].z = -1;
                return;
            }
            
            p[0].x = Dx*t0+x0;
            p[0].y = Dy*t0+y0;
            p[0].z = 1;
            p[1].x = Dx*t1+x0;
            p[1].y = Dy*t1+y0;
            p[1].z = 1;
            return;
        } 
        // only clip p0
        t0 = std::max(std::min(t0,1.0f),0.0f);
        p[0].x = Dx*t0+x0;
        p[0].y = Dy*t0+y0;
        p[0].z = 1;
        p[1].x = x1;
        p[1].y = y1;
        p[1].z = 0;
        return;
    } else {
        float A = (c*c*Dy*Dy+Dx*Dx);
        A = std::max(A,1e-12f);
    
        float B = 2*(c*c*y0*Dy+x0*Dx);
        float D = B*B-4*A*C;
        float sqrtD = sqrtf(std::max(D,0.0f));
        //float t0 = (-B-sqrtD)/(2*A);
        float t1 = (-B+sqrtD)/(2*A);
        
        if (c*c*(y1*y1-1)+x1*x1>0 && x1<0) {// clip p1
            t1 = std::max(std::min(t1,1.0f),0.0f);
            p[1].x = Dx*t1+x0;
            p[1].y = Dy*t1+y0;
            p[1].z = 1;
            p[0].x = x0;
            p[0].y = y0;
            p[0].z = 0;
            return;
        }
    }
    p[0].x = x0;
    p[0].y = y0;
    p[0].z = 0;
    p[1].x = x1;
    p[1].y = y1;
    p[1].z = 0;
    return;
}

inline float int_arc(float y0, float y1) {
    return -0.5*((std::asin(y1)-std::asin(y0))+(y1*std::sqrt(1-y1*y1)-y0*std::sqrt(1-y0*y0)));
}

inline float int_line(float x0, float y0, float x1, float y1, const float c) {
    float Dy = y1-y0;
    float Dx = x1-x0;
    float r = std::sqrt(Dx*Dx+Dy*Dy);
    if (r<1e-6) { //point zero
        return 0.0f;
    }
    Dx/=r;Dy/=r;
    r = Dy*x0-Dx*y0;
    r = std::sqrt(std::max(1-r*r,0.0f));
    
    float p0 = (Dx*x0+Dy*y0)/r;
    float p1 = (Dx*x1+Dy*y1)/r;
    
    float disk = Dy*r*r*(p1*std::sqrt(1-p1*p1)-p0*std::sqrt(1-p0*p0)+std::asin(p1)-std::asin(p0));
    disk = std::abs(disk)*sign(y1-y0);
    float cube = (x1+x0)*(y1-y0);
    return 0.5*(c*cube-std::sqrt(1-c*c)*disk);
}



Float project_triangle(
    Point2 &p0, Point2 &p1, Point2 &p2,
    const float c
) {
    // clip all the edges
    Point3 p[6];// 3 edges in total
    uint32_t front = 0;
    clip_arc(p0, p1, c, p);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    clip_arc(p1, p2, c, p+front*2);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    clip_arc(p2, p0, c, p+front*2);
    if (p[front*2].z>=0) {// not drop current edge
        front += 1;
    }
    
    // eval all integrals
    Float result = 0.0f;
    for (uint32_t i=0; i < front; i++) {
        result += int_line(p[i*2].x, p[i*2].y, p[i*2+1].x, p[i*2+1].y,c); // line integral
        
        if (p[i*2+1].z>0 && p[((i+1)%front)*2].z>0) { // need to integate arc
            result += int_arc(p[i*2+1].y,p[((i+1)%front)*2].y);
        }
    }
    return std::abs(result);
}


/**
 * \brief Implementation of the Beckman and GGX / Trowbridge-Reitz microfacet
 * distributions and various useful sampling routines
 *
 * Based on the papers
 *
 *   "Microfacet Models for Refraction through Rough Surfaces"
 *    by Bruce Walter, Stephen R. Marschner, Hongsong Li, and Kenneth E. Torrance
 *
 * and
 *
 *   "Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals"
 *    by Eric Heitz and Eugene D'Eon
 *
 *  The visible normal sampling code was provided by Eric Heitz and Eugene D'Eon.
 */
 
class GlintDistribution {
public:
    enum Etype {
        EBox = 0,
        EGaussian = 1,
        EDisk = 2,
    };

    /**
     * \brief Create a microfacet distribution from a Property data
     * structure
     */
    GlintDistribution() {}
    
    void initialize(std::string fileName, int32_t res, Float thresh, Float alpha, std::string filter) {
        m_res = res;
        m_thresh = thresh;
        
        if (filter == "box")
            m_type = EBox;
        else if (filter == "gaussian")
            m_type = EGaussian;
        else if (filter == "disk")
            m_type = EDisk;
        else
            SLog(EError, "Specified an invalid filter \"%s\", must be "
                "\"box\",  or \"gaussian\"/\"as\"!", filter.c_str());
        
        m_alphaU = m_alphaV = alpha;
        if (m_alphaU == 0 || m_alphaV == 0) {
            SLog(EWarn, "Cannot create a microfacet distribution with alphaU/alphaV=0 (clamped to 0.0001)."
                    "Please use the corresponding smooth reflectance model to get zero roughness.");
        }
        m_alphaU = std::max(m_alphaU, (Float) 1e-4f);
        m_alphaV = std::max(m_alphaV, (Float) 1e-4f);
        
        
        printf("Reading micro normal...");
        
        m_micro_normal = new float[m_res*m_res*2];
        if (!m_micro_normal) printf("\nCannot allocate memory!\n");
        
        std::FILE *fp = std::fopen((fileName+"/normal").c_str(), "rb");
        std::fread(m_micro_normal,sizeof(float),m_res*m_res*2,fp);
        std::fclose(fp);
        printf("OK\n");
        
        printf("Reading bounds...");
#ifdef GLINT_UNBIASED
        fp = std::fopen((fileName+"/bound2").c_str(),"rb");
#else
        fp = std::fopen((fileName+"/bound").c_str(),"rb");
#endif
        float level_f;
        std::fread(&level_f,sizeof(float),1,fp);
        m_level = std::min((uint32_t)level_f,(uint32_t)8);//no more than 9 level
        for (uint32_t level=0; level<=m_level;level++) {
            uint32_t res = m_res >> level;
            m_bound[level] = new float[res*res*4];
            std::fread(m_bound[level],sizeof(float),res*res*4,fp);
        }
        std::fclose(fp);
        printf("OK\n");
        
        printf("Reading trees...");
        fp = std::fopen((fileName+"/tree").c_str(),"rb");
        for (uint32_t level=0; level<GLINT_TREE_LEVEL;level++) {
            uint32_t res = m_res >> (level+1);
            m_tree[level] = new float[res*res*9];
            std::fread(m_tree[level],sizeof(float),res*res*9,fp);
        }
        std::fclose(fp);
        printf("OK\n");
    }
    
    /// Return the roughness (isotropic case)
    inline Float getAlpha() const { return m_alphaU; }

    /// Return the roughness along the tangent direction
    inline Float getAlphaU() const { return m_alphaU; }

    /// Return the roughness along the bitangent direction
    inline Float getAlphaV() const { return m_alphaV; }
    
    inline bool isIsotropic() const { return false; }
    
    inline Point2 fetch(int32_t i, int32_t j) const {
        i = math::modulo(i,m_res);
        j = math::modulo(j,m_res);
        //i %= m_res;
        //j %= m_res;
        return Point2(m_micro_normal[i*m_res*2+j*2],
                      m_micro_normal[i*m_res*2+j*2+1]);
    }
    
    inline void fetch4(float i, float j, Point2* n) const {
        int32_t i0 = (int)floor(i);
        int32_t j0 = (int)floor(j);
        int32_t i1 = (int)ceil(i);
        int32_t j1 = (int)ceil(j);
        n[0] = fetch(i0,j0);
        n[1] = fetch(i0,j1);
        n[2] = fetch(i1,j0);
        n[3] = fetch(i1,j1);
        return;
    }
    
    inline void fetch4(int32_t i0, int32_t j0, Point2* n) const {
        int32_t i1 = i0+1;
        int32_t j1 = j0+1;
        n[0] = fetch(i0,j0);
        n[1] = fetch(i0,j1);
        n[2] = fetch(i1,j0);
        n[3] = fetch(i1,j1);
        return;
    }
    
    inline Float area3(Point2 &p1, Point2 &p2, Point2 &p3) const {
        return (p1.x-p3.x)*(p2.y-p3.y)
             - (p1.y-p3.y)*(p2.x-p3.x);
    }

    inline Float eval(const Point2 &uv_in, const Float sigmaU, const Float sigmaV, const Vector &m) const {
        if (Frame::cosTheta(m) <= 0)
            return 0.0f;
        Point2 m2(m.x,m.y);
        Point2 uv(fmod(uv_in.x,1.0f)*m_res,fmod(uv_in.y,1.0f)*m_res);
        //Point2 uv(uv_in.x*m_res,uv_in.y*m_res);
        
        // base mip level
        uint32_t idx,res;
        //int32_t u0,v0,u1,v1;
        int32_t u0=-1000,v0=-1000,u1=1000,v1=1000;
        float u,v;
        std::stack<Point3i> record; // (u0,v0,level)
        int32_t level = std::min((int32_t)ceil(log2(std::max(sigmaU,sigmaV))+1),(int32_t)m_level);
        level = std::max(level,0);
        //float thresh_scale = 100*m_thresh;
        float thresh_scale = sigmaU*sigmaV*m_thresh;
        Float pdf=0.0f;
        while (true) {
            res = 1<<level;
            u0 = std::max(u0,(int32_t)floor((uv.x-sigmaU)/res));
            v0 = std::max(v0,(int32_t)floor((uv.y-sigmaV)/res));
            u1 = std::min(u1,((int32_t)ceil((uv.x+sigmaU)/res))-1);
            v1 = std::min(v1,((int32_t)ceil((uv.y+sigmaV)/res))-1);
            
            res = m_res>>level;
            
            // iterate over all possible node
            for (int32_t dv=v0;dv<=v1;dv++) {
                for (int32_t du=u0;du<=u1;du++) {
                    // check whether within the st bound
                    idx= 4*(res*math::modulo(dv,res) 
                               +math::modulo(du,res));
                    u=m_bound[level][idx];
                    v=m_bound[level][idx+1];
                    if ((m2.x<u) || (m2.y<v)) {
                        continue;
                    }
                    u=m_bound[level][idx+2];
                    v=m_bound[level][idx+3];
                    if ((m2.x>u) || (m2.y>v)) {
                        continue;
                    }
                    
                    if (level==0) {
                        // eval triangle
                        
                        Point2 n[4];
                        fetch4(dv,du,n);
                        
                        /**
                        0 - 1
                        | / |
                        2 - 3
                        */
                        float area;
                        v = area3(m2,n[0],n[1]);
                        u = area3(m2,n[2],n[0]);
                        area = area3(n[0],n[1],n[2]);
#ifdef GLINT_UNBIASED
                        if (std::abs(area)<1e-6f) {
                                // ray-eqtri intersect
                                Point2 n0(0+n[1].x, 0.0006204032394013997f+n[1].y);
                                Point2 n1(0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);
                                Point2 n2(-0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);

                                v = area3(m2,n0,n1);
                                u = area3(m2,n2,n0);
                                area = area3(n0,n1,n2);
                        }
#endif

                        area = area>=0 ? 1/std::max(area,1e-6f) 
                                       : 1/std::min(area,-1e-6f);
                        u *= area;
                        v *= area;
                        if ((u>=0)&&(v>=0)&&((u+v)<=1)) {
                            u = (u+du-uv.x);
                            v = (v+dv-uv.y);
                            if ((abs(u)<=sigmaU)&&(abs(v)<=sigmaV)) {
                                switch (m_type) {
                                    case EBox:
                                        pdf += abs(area);
                                        break;
                                    case EGaussian:
                                        pdf += abs(area)*INV_PI*18*std::exp(
                                            -4.5*(u*u/(sigmaU*sigmaU)
                                                 +v*v/(sigmaV*sigmaV))
                                        );
                                        break;
                                    case EDisk:
                                        pdf += abs(area)*INV_PI*4.0f;
                                        break;
                                }
                            }
                        }
                        
                        v = area3(m2,n[2],n[3]);
                        u = area3(m2,n[3],n[1]);
                        area = area3(n[1],n[2],n[3]);
#ifdef GLINT_UNBIASED
                        if (std::abs(area)<1e-6f) {
                                // ray-eqtri intersect
                                Point2 n3(0+n[1].x, 0.0006204032394013997f+n[1].y);
                                Point2 n1(0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);
                                Point2 n2(-0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);

                                v = area3(m2,n2,n3);
                                u = area3(m2,n3,n1);
                                area = area3(n1,n2,n3);
                        }
#endif
                        area = area>=0 ? 1/std::max(area,1e-6f) 
                                       : 1/std::min(area,-1e-6f);
                        u *= area;
                        v *= area;
                        if ((u>=0)&&(v>=0)&&((u+v)<=1)) {
                            u = (1-u)+du-uv.x;
                            v = (1-v)+dv-uv.y;
                            if ((abs(u)<=sigmaU)&&(abs(v)<=sigmaV)) {
                                switch (m_type) {
                                    case EBox:
                                        pdf += abs(area);
                                        break;
                                    case EGaussian:
                                        pdf += abs(area)*INV_PI*18*std::exp(
                                            -4.5*(u*u/(sigmaU*sigmaU)
                                                 +v*v/(sigmaV*sigmaV))
                                        );
                                        break;
                                    case EDisk:
                                        pdf += abs(area)*INV_PI*4.0f;
                                        break;
                                }
                            }
                        }
                    } else if (level <= GLINT_TREE_LEVEL) {
                        idx = (idx/4)*9;
                        float residual = m_tree[level-1][idx];
                        if (residual < thresh_scale) {// can use simplified mesh
                            Point2 n[4];
                            float area;
                            int fac = (1<<level);
                            for (uint32_t k=0; k<4; k++) {
                                n[k].x = m_tree[level-1][idx+k*2+1];
                                n[k].y = m_tree[level-1][idx+k*2+2];
                            }

                            v = area3(m2,n[0],n[1]);
                            u = area3(m2,n[2],n[0]);
                            area = area3(n[0],n[1],n[2]);
#ifdef GLINT_UNBIASED
                            if (std::abs(area)<1e-6f) {
                                // ray-eqtri intersect
                                Point2 n0(0+n[1].x, 0.0006204032394013997f+n[1].y);
                                Point2 n1(0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);
                                Point2 n2(-0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);

                                v = area3(m2,n0,n1);
                                u = area3(m2,n2,n0);
                                area = area3(n0,n1,n2);
                            }
#endif
                            area = area>=0 ? 1/std::max(area,1e-6f) 
                                           : 1/std::min(area,-1e-6f);
                            u *= area;
                            v *= area;
                            if ((u>=0)&&(v>=0)&&((u+v)<=1)) {
                                u = (fac*(u+du)-uv.x);
                                v = (fac*(v+dv)-uv.y);
                                if ((abs(u)<=sigmaU)&&(abs(v)<=sigmaV)) {
                                    switch (m_type) {
                                        case EBox:
                                            pdf += abs(area)*fac*fac;
                                            break;
                                        case EGaussian:
                                            pdf += abs(area)*fac*fac
                                                  *INV_PI*18*std::exp(
                                                -4.5*(u*u/(sigmaU*sigmaU)
                                                     +v*v/(sigmaV*sigmaV))
                                            );
                                            break;
                                        case EDisk:
                                            pdf += abs(area)*fac*fac*INV_PI*4.0f;
                                            break;
                                    }
                                }
                            }

                            v = area3(m2,n[2],n[3]);
                            u = area3(m2,n[3],n[1]);
                            area = area3(n[1],n[2],n[3]);
#ifdef GLINT_UNBIASED
                            if (std::abs(area)<1e-6f) {
                                    // ray-eqtri intersect
                                    Point2 n3(0+n[1].x, 0.0006204032394013997f+n[1].y);
                                    Point2 n1(0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);
                                    Point2 n2(-0.000537284965911771f+n[1].x,-0.00031020161970069985f+n[1].y);

                                    v = area3(m2,n2,n3);
                                    u = area3(m2,n3,n1);
                                    area = area3(n1,n2,n3);
                            }
#endif
                            area = area>=0 ? 1/std::max(area,1e-6f) 
                                           : 1/std::min(area,-1e-6f);
                            u *= area;
                            v *= area;
                            if ((u>=0)&&(v>=0)&&((u+v)<=1)) {
                                u = fac*((1-u)+du)-uv.x;
                                v = fac*((1-v)+dv)-uv.y;
                                if ((abs(u)<=sigmaU)&&(abs(v)<=sigmaV)) {
                                    switch (m_type) {
                                        case EBox:
                                            pdf += abs(area)*fac*fac;
                                            break;
                                        case EGaussian:
                                            pdf += abs(area)*fac*fac
                                                  *INV_PI*18*std::exp(
                                                -4.5*(u*u/(sigmaU*sigmaU)
                                                     +v*v/(sigmaV*sigmaV))
                                            );
                                            break;
                                        case EDisk:
                                            pdf += abs(area)*fac*fac*INV_PI*4.0f;
                                            break;
                                    }
                                }
                            }
                        } else {
                            record.push(Point3i(du*2,dv*2,level-1));
                        } 
                    } else {
                        // add current node to the stack
                        record.push(Point3i(du*2,dv*2,level-1));
                    }
                }
            }
            
            if (record.empty()) {
                break;
            } else {
                Point3i top_record = record.top();
                u0 = top_record.x;
                v0 = top_record.y;
                u1 = u0+1;
                v1 = v0+1;
                level = top_record.z;
                record.pop();
            }
        }
        pdf /= ((sigmaU*2)*(sigmaV*2));
        
        /* Prevent potential numerical issues in other stages of the model */
        if (pdf*Frame::cosTheta(m) < 1e-20f)
            pdf = 0;
        
        return pdf;
    }
    
    inline Float eval(const Vector &m, const Float alphaU, const Float alphaV) const {
        if (Frame::cosTheta(m) <= 0)
            return 0.0f;

        Float cosTheta2 = Frame::cosTheta2(m);
        Float beckmannExponent = ((m.x*m.x) / (alphaU * alphaU)
                + (m.y*m.y) / (alphaV * alphaV)) / cosTheta2;
        Float root = ((Float) 1 + beckmannExponent) * cosTheta2;
        Float result = (Float) 1 / (M_PI * alphaU * alphaV * root * root);
        /* Prevent potential numerical issues in other stages of the model */
        if (result * Frame::cosTheta(m) < 1e-20f)
            result = 0;
        return result;
    }


    inline Normal sample_gndf(const Point2 &uv_in, const Float sigmaU, const Float sigmaV, const Point2 &sample) const {
        float x = fmod(uv_in.x,1.0f)*m_res;
        float y = fmod(uv_in.y,1.0f)*m_res;
        float r;
        switch (m_type) {
            case EBox:
                x += sigmaU*(sample.x*2-1);
                y += sigmaV*(sample.y*2-1);
                break;
            case EGaussian:
                r = std::sqrt(-2.0f*std::log(math::clamp(sample.x,1e-20f,1.0f)));
                x += r*std::cos(2*M_PI*sample.y)*sigmaU/3.0f;
                y += r*std::sin(2*M_PI*sample.y)*sigmaV/3.0f;
                break;
            case EDisk:
                r = std::sqrt(sample.x);
                x += r*sigmaU*std::cos(2*M_PI*sample.y);
                y += r*sigmaV*std::sin(2*M_PI*sample.y);
                break;
        }
        //float x = fmod(uv_in.x,1.0f)*m_res+sigmaU*(sample.x*2-1);
        //float y = fmod(uv_in.y,1.0f)*m_res+sigmaV*(sample.y*2-1);
        //float x = uv_in.x*m_res+sigmaU*(sample.x*2-1);
        //float y = uv_in.y*m_res+sigmaV*(sample.y*2-1);
        
#ifdef GLINT_UNBIASED
        int32_t level = std::min((int32_t)ceil(log2(std::max(sigmaU,sigmaV))+1),GLINT_TREE_LEVEL);
        level = std::max(level,0);
        float thresh_scale = sigmaU*sigmaV*m_thresh;
        uint32_t res,idx;
        res = 1<<level;
        x/=res;
        y/=res;
        res = m_res>>level;
        int32_t u0,v0;
        Point2 n[4];

        while (true) {
            if (level==0) {
                fetch4(y,x,n);
                x -= floor(x);
                y -= floor(y);
                break;
            }
            u0 = (int32_t)floor(x);
            v0 = (int32_t)floor(y);
            idx= 9*(res*math::modulo(v0,res) 
               + math::modulo(u0,res));
            float residual = m_tree[level-1][idx];
            if (residual < thresh_scale) {// can use simplified mesh        
                for (uint32_t k=0; k<4; k++) {
                    n[k].x = m_tree[level-1][idx+k*2+1];
                    n[k].y = m_tree[level-1][idx+k*2+2];
                }
                x -= u0;
                y -= v0; 
                break;
            } else {
                x*=2;
                y*=2;
                res*=2;
                level -= 1;
            }
        }
        
        Point2 st;
        if (x+y<1) { // upper triangle
            Float area = area3(n[0],n[1],n[2]);
            if (std::abs(area)<1e-6f) {
                // free sample from x,y
                st = n[1]
                   + 
                   Point2(
                    0.000537284965911771f*x - 0.000537284965911771f*y,
                    -0.0009306048591021f*x - 0.0009306048591021f*y + 0.0006204032394014f
                   );
            } else {
                st = n[0]*(1-x-y) + n[1]*x + n[2]*y;
            }
        } else {
            Float area = area3(n[2],n[1],n[3]);
            if (std::abs(area)<1e-6f) {
                st = n[1]
                   +  
                   Point2(
                    0.000537284965911771f*x - 0.000537284965911771f*y,
                    0.0009306048591021f*x + 0.0009306048591021f*y - 0.0012408064788028f
                   );
            } else {
                st = n[2]*(1-x) + n[1]*(1-y) + n[3]*(x+y-1);
            }
        }
#else
        Point2 n[4];
        fetch4(y,x,n);
        x -= floor(x);
        y -= floor(y);
        Point2 st;
        if (x+y<1) { // upper triangle
            st = n[0]*(1-x-y) + n[1]*x + n[2]*y;
        } else {
            st = n[2]*(1-x) + n[1]*(1-y) + n[3]*(x+y-1);
        }
#endif
        return Normal(st.x,st.y,std::sqrt(1.0f-st.x*st.x-st.y*st.y));
    }

    /**
     * \brief Wrapper function which calls \ref sampleAll() or \ref sampleVisible()
     * depending on the parameters of this class
     */
    inline Normal sample(const Point2 &uv_in, const Float sigmaU, const Float sigmaV, const Point2 &sample, Float &pdf) const {
        Normal m = sample_gndf(uv_in, sigmaU, sigmaV, sample);
        pdf = eval(uv_in, sigmaU, sigmaV, m)*Frame::cosTheta(m);
        
        return m;
    }

    /**
     * \brief Wrapper function which calls \ref sampleAll() or \ref sampleVisible()
     * depending on the parameters of this class
     */
    inline Normal sample(const Point2 &uv_in, const Float sigmaU, const Float sigmaV, const Point2 &sample) const {
        Normal m = sample_gndf(uv_in, sigmaU, sigmaV, sample);
        return m;
    }
    
    inline Normal sample(const Point2 &sample, Float &pdf, const Float alphaU, const Float alphaV) const {
        Normal m;
        Float cosThetaM = 0.0f;
        Float sinPhiM, cosPhiM;
        Float alphaSqr;
        
        /* Sample phi component (anisotropic case) */
        Float phiM = std::atan(alphaV / alphaU *
                    std::tan(M_PI + 2*M_PI*sample.y)) + M_PI * std::floor(2*sample.y + 0.5f);
        math::sincos(phiM, &sinPhiM, &cosPhiM);
        Float cosSc = cosPhiM / alphaU, sinSc = sinPhiM / alphaV;
        alphaSqr = 1.0f / (cosSc*cosSc + sinSc*sinSc);
        
        /* Sample theta component */
        Float tanThetaMSqr = alphaSqr * sample.x / (1.0f - sample.x);
        cosThetaM = 1.0f / std::sqrt(1.0f + tanThetaMSqr);

        /* Compute probability density of the sampled position */
        Float temp = 1+tanThetaMSqr/alphaSqr;
        pdf = INV_PI / (alphaU*alphaV*cosThetaM*cosThetaM*cosThetaM*temp*temp);
        
        /* Prevent potential numerical issues in other stages of the model */
        if (pdf < 1e-20f)
            pdf = 0;

        Float sinThetaM = std::sqrt(
            std::max((Float) 0, 1 - cosThetaM*cosThetaM));

        return Vector(
            sinThetaM * cosPhiM,
            sinThetaM * sinPhiM,
            cosThetaM
        );
    }
    

    /**
     * \brief Wrapper function which calls \ref pdfAll() or \ref pdfVisible()
     * depending on the parameters of this class
     */
    inline Float pdf(const Point2 &uv_in, const Float sigmaU, const Float sigmaV, const Vector &m) const {
        return eval(uv_in, sigmaU, sigmaV, m)*Frame::cosTheta(m);
    }

    inline Float pdf(const Vector &m, const Float alphaU, const Float alphaV) const {
        return eval(m, alphaU, alphaV)*Frame::cosTheta(m);
    }


    /**
     * \brief Smith's shadowing-masking function G1 for each
     * of the supported microfacet distributions
     *
     * \param v
     *     An arbitrary direction
     * \param m
     *     The microfacet normal
     */
    Float smithG1(const Vector &v, const Vector &m) const {
        /* Ensure consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
        if (dot(v, m) * Frame::cosTheta(v) <= 0)
            return 0.0f;

        /* Perpendicular incidence -- no shadowing/masking */
        Float tanTheta = std::abs(Frame::tanTheta(v));
        if (tanTheta == 0.0f)
            return 1.0f;

        Float alpha = projectRoughness(v);
        Float root = alpha * tanTheta;
        return 2.0f / (1.0f + math::hypot2((Float) 1.0f, root));
    }
    
    Float smithG1(const Vector &v, const Vector &m, const Float alphaU, const Float alphaV) const {
        /* Ensure consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
        if (dot(v, m) * Frame::cosTheta(v) <= 0)
            return 0.0f;

        /* Perpendicular incidence -- no shadowing/masking */
        Float tanTheta = std::abs(Frame::tanTheta(v));
        if (tanTheta == 0.0f)
            return 1.0f;

        Float alpha = projectRoughness(v,alphaU,alphaV);
        Float root = alpha * tanTheta;
        return 2.0f / (1.0f + math::hypot2((Float) 1.0f, root));
    }

    /**
     * \brief Separable shadow-masking function based on Smith's
     * one-dimensional masking model
     */
    inline Float G(const Vector &wi, const Vector &wo, const Vector &m) const {
        return smithG1(wi, m) * smithG1(wo, m);
    }
    
    inline Float G(const Vector &wi, const Vector &wo, const Vector &m, const Float alphaU, const Float alphaV) const {
        return smithG1(wi, m, alphaU, alphaV) * smithG1(wo, m, alphaU, alphaV);
    }
    
    
    Float G(const Point2 &uv_in, const Float sigmaU, const Float sigmaV,
        const Vector wi, const Vector wo, const Vector &wm 
    ) const {
        if (dot(wi, wm) * Frame::cosTheta(wi) <= 0)
            return 0.0f;
        if (dot(wo, wm) * Frame::cosTheta(wo) <= 0)
            return 0.0f;
        
        Point2 uv(fmod(uv_in.x,1.0f)*m_res,fmod(uv_in.y,1.0f)*m_res);
        
        // base mip level
        int32_t u0 = (int32_t)floor((uv.x-sigmaU));
        int32_t v0 = (int32_t)floor((uv.y-sigmaV));
        int32_t u1 = ((int32_t)ceil((uv.x+sigmaU)))-1;
        int32_t v1 = ((int32_t)ceil((uv.y+sigmaV)))-1;
        
        float phi_i = 0.0f;
        float phi_o = 0.0f;
        if (wi.z < 0.99999f) {
            phi_i = std::atan2(wi.y,wi.x);
            phi_o = std::atan2(wo.y,wo.x);
        }
        Vector2 To,Ti;
        math::sincos(phi_i, &Ti.y, &Ti.x);
        math::sincos(phi_o, &To.y, &To.x);
        
        float Pi=0.0f;
        float Po=0.0f;
        Point2 n[4];
        Point2 m[4];
        for (int32_t dv=v0;dv<=v1;dv++) {
            for (int32_t du=u0;du<=u1;du++) {
                fetch4(dv,du,n);
                Float area0 = abs(area3(n[0],n[1],n[2]));
                Float area1 = abs(area3(n[2],n[1],n[3]));
                Float nz = std::sqrt(std::max(1-n[1].x*n[1].x-n[1].y*n[1].y,0.0f));
                
                // project wi
                for (int i=0; i<4; i++) {
                    m[i].x = Ti.x*n[i].x+Ti.y*n[i].y;
                    m[i].y = Ti.x*n[i].y-Ti.y*n[i].x;
                }
                if (area0>1e-6) {
                    Pi += project_triangle(m[0],m[1],m[2],wi.z)/area0;
                } else {
                    Pi += 0.5f*std::max(wi.x*n[1].x+wi.y*n[1].y+wi.z*nz,0.0f);
                }
                if (area1>1e-6) {
                    Pi += project_triangle(m[2],m[1],m[3],wi.z)/area1;
                } else {
                    Pi += 0.5f*std::max(wi.x*n[1].x+wi.y*n[1].y+wi.z*nz,0.0f);
                }
                
                // project wo
                for (int i=0; i<4; i++) {
                    m[i].x = To.x*n[i].x+To.y*n[i].y;
                    m[i].y = To.x*n[i].y-To.y*n[i].x;
                }
                if (area0>1e-6) {
                    Po += project_triangle(m[0],m[1],m[2],wo.z)/area0;
                } else {
                    Po += 0.5f*std::max(wo.x*n[1].x+wo.y*n[1].y+wo.z*nz,0.0f);
                }
                if (area1>1e-6) {
                    Po += project_triangle(m[2],m[1],m[3],wo.z)/area1;
                } else {
                    Po += 0.5f*std::max(wo.x*n[1].x+wo.y*n[1].y+wo.z*nz,0.0f);
                }
            }
        }
        
        Pi /= (v1-v0+1)*(u1-u0+1);
        Po /= (v1-v0+1)*(u1-u0+1);
        
        Pi = std::min(std::max(Pi,Frame::cosTheta(wi)),1.0f);
        Po = std::min(std::max(Po,Frame::cosTheta(wo)),1.0f);

        if (Pi*Po == 0) {
            return 0.0f;
        }
        
        return Frame::cosTheta(wi)*Frame::cosTheta(wo)/(Pi*Po);
    }

    /// Return a string representation of the contents of this instance
    std::string toString() const {
        return formatString("GlintDistribution[alphaU=%f, alphaV=%f]",
            m_alphaU, m_alphaV);
    }
protected:
    /// Compute the effective roughness projected on direction \c v
    inline Float projectRoughness(const Vector &v) const {
        Float invSinTheta2 = 1 / Frame::sinTheta2(v);

        if (isIsotropic() || invSinTheta2 <= 0)
            return m_alphaU;

        Float cosPhi2 = v.x * v.x * invSinTheta2;
        Float sinPhi2 = v.y * v.y * invSinTheta2;

        return std::sqrt(cosPhi2 * m_alphaU * m_alphaU + sinPhi2 * m_alphaV * m_alphaV);
    }
    
    inline Float projectRoughness(const Vector &v, const Float alphaU, const Float alphaV) const {
        Float invSinTheta2 = 1 / Frame::sinTheta2(v);

        if (isIsotropic() || invSinTheta2 <= 0)
            return alphaU;

        Float cosPhi2 = v.x * v.x * invSinTheta2;
        Float sinPhi2 = v.y * v.y * invSinTheta2;

        return std::sqrt(cosPhi2 * alphaU * alphaU + sinPhi2 * alphaV * alphaV);
    }

protected:
    Float m_alphaU, m_alphaV;
    Float m_thresh;
    float* m_micro_normal;
    float* m_bound[9];
    float* m_tree[GLINT_TREE_LEVEL];
    uint32_t m_level;
    uint32_t m_res;
    Etype m_type;
};
MTS_NAMESPACE_END

#endif /* __GLINT_H */
